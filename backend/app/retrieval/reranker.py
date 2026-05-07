"""
Cross-encoder re-ranker.

The hybrid search bi-encoder ranks chunks by independent query and chunk
embeddings — it never sees them together. A cross-encoder (ms-marco)
reads the query and chunk concatenated, producing a more accurate
relevance score at the cost of O(n) forward passes.

We apply it to the top-K pool from multi_query_retrieve and keep only
the highest-scoring chunks for the final prompt.
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
    Re-score and re-rank chunks using a cross-encoder.

    Args:
        query:  user query string
        chunks: candidate chunks from multi_query_retrieve (any length)
        top_k:  number of chunks to keep after re-ranking

    Returns:
        top_k chunks sorted by cross-encoder score descending, with
        a "rerank_score" key added to each chunk dict.
    """
    k = top_k or settings.rerank_top_k

    if not chunks:
        return []

    model = _get_model()
    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores: list[float] = model.predict(pairs).tolist()

    scored = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        {**chunk, "rerank_score": score}
        for score, chunk in scored[:k]
    ]
