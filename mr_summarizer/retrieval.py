"""Hybrid retrieval: three retrievers -> Reciprocal Rank Fusion -> cross-encoder rerank.

    question
       |-- keyword search   (Postgres full-text: exact terms, tickers, names)
       |-- vector search    (pgvector: meaning, paraphrases)
       |-- graph search     (knowledge graph: entities in the question + their neighbours)
       v
    RRF fusion -> top 20 -> cross-encoder rerank -> top 6 chunks for the LLM
"""
from dataclasses import dataclass, field
from functools import lru_cache

from . import graph, models
from .config import settings
from .db import connect
from .fusion import rrf


@dataclass
class Hit:
    chunk_id: int
    content: str
    title: str
    url: str | None
    rrf_score: float
    rerank_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass
class Retrieval:
    hits: list[Hit]
    entities: list[str]                # entities recognised in the question
    ranked_lists: dict[str, list[int]]  # raw output of each retriever, for debugging


# ---------- the three retrievers ----------

def keyword_search(cur, query: str, n: int) -> list[int]:
    # plainto_tsquery ANDs every word, which matches almost nothing for a full
    # question. Swapping & for | turns it into an OR query; ts_rank_cd then rewards
    # chunks that contain more of the terms, closer together.
    cur.execute(
        """WITH q AS (
               SELECT NULLIF(replace(plainto_tsquery('english', %s)::text, '&', '|'), '')::tsquery AS q
           )
           SELECT c.id FROM chunks c, q
           WHERE q.q IS NOT NULL AND c.tsv @@ q.q
           ORDER BY ts_rank_cd(c.tsv, q.q) DESC
           LIMIT %s""",
        (query, n),
    )
    return [r[0] for r in cur.fetchall()]


def vector_search(cur, query: str, n: int) -> list[int]:
    # <=> is pgvector's cosine distance; the HNSW index makes this fast
    cur.execute(
        "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (models.embed_query(query), n),
    )
    return [r[0] for r in cur.fetchall()]


@lru_cache(maxsize=1)
def _entity_keys() -> dict[str, list[int]]:
    """Every entity key in the graph -> node ids. Cached; restart the API after ingesting."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT key, id FROM entities WHERE length(key) >= 3")
        keys: dict[str, list[int]] = {}
        for key, entity_id in cur.fetchall():
            keys.setdefault(key, []).append(entity_id)
        return keys


def graph_search(cur, query: str, n: int) -> tuple[list[int], list[str]]:
    """Score chunks by the entities they mention.

    Seeds are entities named in the question (worth 2 points per chunk).
    Neighbours are the 15 entities most strongly connected to the seeds
    (worth 0.5 each), which pulls in related context the question did not name,
    e.g. a question about a company also surfaces its CEO and competitors.
    """
    seeds = graph.entities_in_query(query, _entity_keys())
    if not seeds:
        return [], []
    cur.execute(
        """WITH seeds AS (SELECT unnest(%(seeds)s::int[]) AS id),
           nbrs AS (
               SELECT CASE WHEN e.src = s.id THEN e.dst ELSE e.src END AS id, e.weight
               FROM entity_edges e JOIN seeds s ON s.id IN (e.src, e.dst)
               ORDER BY e.weight DESC
               LIMIT 15
           ),
           scored AS (
               SELECT ce.chunk_id,
                      SUM(CASE WHEN ce.entity_id IN (SELECT id FROM seeds) THEN 2.0 ELSE 0.5 END) AS score
               FROM chunk_entities ce
               WHERE ce.entity_id IN (SELECT id FROM seeds UNION SELECT id FROM nbrs)
               GROUP BY ce.chunk_id
           )
           SELECT chunk_id FROM scored ORDER BY score DESC, chunk_id DESC LIMIT %(n)s""",
        {"seeds": seeds, "n": n},
    )
    ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT DISTINCT name FROM entities WHERE id = ANY(%s)", (seeds,))
    return ids, [r[0] for r in cur.fetchall()]


# ---------- putting it together ----------

def retrieve(query: str, use_rerank: bool = True) -> Retrieval:
    n = settings.candidates_per_retriever
    with connect() as conn, conn.cursor() as cur:
        graph_ids, entities = graph_search(cur, query, n)
        ranked = {
            "keyword": keyword_search(cur, query, n),
            "vector": vector_search(cur, query, n),
            "graph": graph_ids,
        }
        fused = rrf(ranked, k=settings.rrf_k)[: settings.rerank_top_n]
        if not fused:
            return Retrieval([], entities, ranked)

        cur.execute(
            """SELECT c.id, c.content, a.title, a.url
               FROM chunks c JOIN articles a ON a.id = c.article_id
               WHERE c.id = ANY(%s)""",
            ([f.chunk_id for f in fused],),
        )
        rows = {r[0]: r for r in cur.fetchall()}

    hits = [
        Hit(f.chunk_id, rows[f.chunk_id][1], rows[f.chunk_id][2], rows[f.chunk_id][3], f.score, ranks=f.ranks)
        for f in fused
        if f.chunk_id in rows
    ]
    if use_rerank:
        for hit, score in zip(hits, models.rerank_scores(query, [h.content for h in hits])):
            hit.rerank_score = score
        hits.sort(key=lambda h: h.rerank_score, reverse=True)

    return Retrieval(hits[: settings.final_top_k], entities, ranked)
