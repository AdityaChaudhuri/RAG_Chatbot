"""
Local embedding using BAAI/bge-base-en-v1.5 via sentence-transformers.

Runs entirely on the server — no API key, no rate limits, no cost per call.
Output dimension is 768, matching VECTOR(768) in the database schema.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None
_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_BATCH_SIZE = 32  # conservative default — safe on CPU with limited RAM

# BGE models are trained with this prefix on the query side.
# Omitting it at query time degrades retrieval accuracy by ~2-3%.
# Documents are embedded without any prefix.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def embed_documents(texts: list[str]) -> list[list[float]]:
    embeddings: np.ndarray = _get_model().encode(
        texts,
        batch_size=_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single user query with the BGE query prefix."""
    embedding: np.ndarray = _get_model().encode(
        [_QUERY_PREFIX + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding[0].tolist()
