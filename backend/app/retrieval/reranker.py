"""
Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2).

A cross-encoder reads the query and chunk together as a single input and outputs
one relevance score — more accurate than the bi-encoder used for retrieval,
at the cost of one forward pass per candidate. We only apply it to the top-K pool
from multi_query_retrieve, not to all chunks.
"""

from sentence_transformers import CrossEncoder

from app.config import settings

_MODEL: CrossEncoder | None = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> CrossEncoder:
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(_MODEL_NAME, max_length=512)
    return _MODEL


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Re-score and re-rank chunks using the cross-encoder.

    Returns top_k chunks sorted by cross-encoder score descending,
    with a "rerank_score" key added to each chunk dict.
    """
    k = top_k or settings.rerank_top_k

    if not chunks:
        return []

    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores: list[float] = _get_model().predict(pairs).tolist()

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    return [
        {**chunk, "rerank_score": score}
        for score, chunk in scored[:k]
    ]
