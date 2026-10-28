"""Load financial news, chunk it, embed it, and build the knowledge graph.

Usage:
    python -m mr_summarizer.ingest --limit 3000
    python -m mr_summarizer.ingest --file my_articles.jsonl   # needs title + text fields

3,000 articles takes roughly 10-20 minutes on a laptop CPU, mostly embedding time.
Re-running is safe: articles already in the database are skipped.
"""
import argparse
import json
import time
from collections import Counter
from typing import Iterator

from . import graph, models, text
from .db import connect

HF_DATASET = "BEE-spoke-data/financial-news-articles-filtered"


def load_articles(limit: int, path: str | None) -> Iterator[dict]:
    if path:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    return
                yield json.loads(line)
    else:
        from datasets import load_dataset

        # streaming=True downloads only the rows we actually read
        ds = load_dataset(HF_DATASET, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= limit:
                return
            yield row


def insert_article(cur, row: dict) -> int | None:
    body = text.clean(row.get("text") or "")
    title = (row.get("title") or "").strip()
    if len(body.split()) < 50 or not title:
        return None  # too short to be useful
    source_id = row.get("url") or title
    cur.execute(
        """INSERT INTO articles (source_id, title, url, body) VALUES (%s, %s, %s, %s)
           ON CONFLICT (source_id) DO NOTHING RETURNING id""",
        (source_id, title, row.get("url"), body),
    )
    hit = cur.fetchone()
    return hit[0] if hit else None


def ingest(limit: int, path: str | None, batch: int = 200) -> None:
    started = time.time()
    n_articles = n_chunks = 0

    with connect() as conn, conn.cursor() as cur:
        pending: list[tuple[int, str]] = []  # (article_id, chunk text) waiting to embed

        def flush():
            nonlocal n_chunks
            if not pending:
                return
            contents = [c for _, c in pending]
            vectors = models.embed_documents(contents)
            # chunk_index restarts per article (an article's chunks always share a batch)
            counters: Counter = Counter()
            params = []
            for (article_id, content), vec in zip(pending, vectors):
                params.append((article_id, counters[article_id], content, vec))
                counters[article_id] += 1
            # executemany with returning=True sends the whole batch in one pipeline,
            # which matters a lot when the database is in the cloud
            cur.executemany(
                """INSERT INTO chunks (article_id, chunk_index, content, embedding)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                params,
                returning=True,
            )
            chunk_ids = []
            while True:
                chunk_ids.append(cur.fetchone()[0])
                if not cur.nextset():
                    break
            mentions = graph.extract_mentions(models.ner().pipe(contents, batch_size=32))
            save_graph(cur, chunk_ids, mentions)
            conn.commit()
            n_chunks += len(pending)
            pending.clear()
            print(f"  {n_articles} articles, {n_chunks} chunks, {time.time() - started:.0f}s")

        for row in load_articles(limit, path):
            article_id = insert_article(cur, row)
            if article_id is None:
                continue
            n_articles += 1
            body = text.clean(row["text"])
            pending.extend((article_id, c) for c in text.chunk(body))
            if len(pending) >= batch:
                flush()
        flush()

    print(f"Done: {n_articles} new articles, {n_chunks} chunks in {time.time() - started:.0f}s")


def save_graph(cur, chunk_ids: list[int], mentions: list[list[tuple[str, str, str]]]) -> None:
    """Upsert entity nodes, link them to chunks, and add or strengthen edges.

    Each step is a single SQL statement over arrays (unnest), so a batch of
    hundreds of chunks costs three round trips to the database, not thousands.
    mentions[i] holds (key, display name, label) tuples for chunk_ids[i].
    """
    # 1. nodes: one row per distinct (key, label), keeping the first display name seen
    nodes: dict[tuple[str, str], str] = {}
    for chunk in mentions:
        for key, name, label in chunk:
            nodes.setdefault((key, label), name)
    if not nodes:
        return
    keys, labels = zip(*nodes)
    cur.execute(
        """INSERT INTO entities (key, name, label)
           SELECT * FROM unnest(%s::text[], %s::text[], %s::text[])
           ON CONFLICT (key, label) DO UPDATE SET key = EXCLUDED.key
           RETURNING id, key, label""",
        (list(keys), list(nodes.values()), list(labels)),
    )
    node_id = {(key, label): i for i, key, label in cur.fetchall()}

    # 2. chunk -> entity links
    per_chunk = [{node_id[(k, l)] for k, _, l in chunk} for chunk in mentions]
    links = [(c, e) for c, ids in zip(chunk_ids, per_chunk) for e in ids]
    if links:
        cs, es = zip(*links)
        cur.execute(
            """INSERT INTO chunk_entities (chunk_id, entity_id)
               SELECT * FROM unnest(%s::int[], %s::int[]) ON CONFLICT DO NOTHING""",
            (list(cs), list(es)),
        )

    # 3. co-occurrence edges, already summed per pair so each pair appears once
    edges = graph.cooccurrence_edges(per_chunk)
    if edges:
        srcs, dsts = zip(*edges)
        cur.execute(
            """INSERT INTO entity_edges (src, dst, weight)
               SELECT * FROM unnest(%s::int[], %s::int[], %s::int[])
               ON CONFLICT (src, dst) DO UPDATE SET weight = entity_edges.weight + EXCLUDED.weight""",
            (list(srcs), list(dsts), list(edges.values())),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=3000, help="max articles to read")
    p.add_argument("--file", help="JSONL file with title/text/url fields instead of the HF dataset")
    args = p.parse_args()
    ingest(args.limit, args.file)
