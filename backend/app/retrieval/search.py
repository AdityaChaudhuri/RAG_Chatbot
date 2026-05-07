"""
Hybrid search client.

Calls the hybrid_search stored procedure defined in 003_procedures.sql.
The procedure internally runs pgvector ANN + PostgreSQL FTS and merges
them with Reciprocal Rank Fusion — this module just marshals the call.
"""

from app.db.client import supabase
from app.ingestion.embedder import embed_query
from app.config import settings


def hybrid_search(
    query: str,
    user_id: str,
    document_id: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """
    Run hybrid retrieval for a user query.

    Args:
        query:       raw user query string
        user_id:     restricts results to this user's chunks (RLS layer 2)
        document_id: optional — scope search to a single document
        top_k:       number of chunks to return before re-ranking

    Returns:
        List of chunk dicts with chunk_id, content, metadata, rrf_score
    """
    k = top_k or settings.retrieval_top_k
    query_vector = embed_query(query)

    result = supabase.rpc(
        "hybrid_search",
        {
            "query_embedding": query_vector,
            "query_text": query,
            "target_user_id": user_id,
            "target_doc_id": document_id,
            "top_k": k,
        },
    ).execute()

    return result.data or []
