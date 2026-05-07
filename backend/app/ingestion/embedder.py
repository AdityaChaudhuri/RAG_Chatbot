"""
Voyage AI embedding client.

voyage-3 produces 1024-dimensional vectors — matching the VECTOR(1024)
column definition in 001_schema.sql.

Two input types are used:
  "document" — for chunks at ingestion time (optimised for retrieval)
  "query"    — for user queries at search time (optimised for matching)

Voyage API has a batch limit of 128 texts per request; embed_documents
handles batching automatically.
"""

import voyageai

from app.config import settings

_CLIENT: voyageai.Client | None = None
_MODEL = "voyage-3"
_BATCH_SIZE = 128


def _get_client() -> voyageai.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = voyageai.Client(api_key=settings.voyage_api_key)
    return _CLIENT


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document chunks. Batches automatically."""
    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        result = client.embed(batch, model=_MODEL, input_type="document")
        all_embeddings.extend(result.embeddings)

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single user query."""
    result = _get_client().embed([text], model=_MODEL, input_type="query")
    return result.embeddings[0]
