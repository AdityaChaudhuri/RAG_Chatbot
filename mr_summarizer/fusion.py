"""Reciprocal Rank Fusion (RRF).

Each retriever returns a ranked list of chunk ids. Their raw scores are not
comparable (BM25-style keyword scores, cosine distances and graph counts live on
different scales), so RRF ignores scores and uses only rank positions:

    rrf(d) = sum over retrievers r of  1 / (k + rank_r(d))

A chunk ranked highly by several retrievers rises to the top. k (usually 60)
dampens the advantage of being ranked 1st versus 5th by a single retriever.
"""
from dataclasses import dataclass, field


@dataclass
class Fused:
    chunk_id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever name -> 1-based rank


def rrf(ranked_lists: dict[str, list[int]], k: int = 60) -> list[Fused]:
    fused: dict[int, Fused] = {}
    for name, ids in ranked_lists.items():
        seen: set[int] = set()
        for rank, chunk_id in enumerate(ids, start=1):
            if chunk_id in seen:  # ignore duplicates within one list
                continue
            seen.add(chunk_id)
            item = fused.setdefault(chunk_id, Fused(chunk_id, 0.0))
            item.score += 1.0 / (k + rank)
            item.ranks[name] = rank
    # ties broken by number of retrievers that agreed, then by id for determinism
    return sorted(fused.values(), key=lambda f: (-f.score, -len(f.ranks), f.chunk_id))
