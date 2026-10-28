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


def ingest(limit: int, path: str | None, batch: int = 100) -> None:
    started = time.time()
    n_articles = n_chunks = 0

    with connect() as conn, conn.cursor() as cur:
        pending: list[tuple[int, str]] = []  # (article_id, chunk text) waiting to embed

        def flush():
            nonlocal n_chunks
            if not pending:
                return
            vectors = models.embed_documents([c for _, c in pending])
            # chunk_index restarts per article
            counters: Counter = Counter()
            chunk_ids: list[int] = []
            for (article_id, content), vec in zip(pending, vectors):
                cur.execute(
                    """INSERT INTO chunks (article_id, chunk_index, content, embedding)
                       VALUES (%s, %s, %s, %s) RETURNING id""",
                    (article_id, counters[article_id], content, vec),
                )
                counters[article_id] += 1
                chunk_ids.append(cur.fetchone()[0])
            add_entities(cur, chunk_ids, [c for _, c in pending])
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


def add_entities(cur, chunk_ids: list[int], contents: list[str]) -> None:
    """Run NER over a batch of chunks, upsert entity nodes, link them to chunks,
    and add or strengthen co-occurrence edges."""
    per_chunk: list[set[int]] = []
    for chunk_id, doc in zip(chunk_ids, models.ner().pipe(contents, batch_size=32)):
        ids: set[int] = set()
        for ent in doc.ents:
            key = graph.normalise(ent.text)
            if not graph.keep(key, ent.label_):
                continue
            cur.execute(
                """INSERT INTO entities (key, name, label) VALUES (%s, %s, %s)
                   ON CONFLICT (key, label) DO UPDATE SET key = EXCLUDED.key
                   RETURNING id""",
                (key, ent.text.strip(), ent.label_),
            )
            ids.add(cur.fetchone()[0])
        for entity_id in ids:
            cur.execute(
                "INSERT INTO chunk_entities VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (chunk_id, entity_id),
            )
        per_chunk.append(ids)

    for (a, b), w in graph.cooccurrence_edges(per_chunk).items():
        cur.execute(
            """INSERT INTO entity_edges (src, dst, weight) VALUES (%s, %s, %s)
               ON CONFLICT (src, dst) DO UPDATE SET weight = entity_edges.weight + EXCLUDED.weight""",
            (a, b, w),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=3000, help="max articles to read")
    p.add_argument("--file", help="JSONL file with title/text/url fields instead of the HF dataset")
    args = p.parse_args()
    ingest(args.limit, args.file)
