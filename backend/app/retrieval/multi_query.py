"""
Multi-query retrieval.

A single user question may not surface all relevant chunks if the phrasing
doesn't match the document's vocabulary. This module asks Claude to generate
N semantically varied reformulations of the query, runs hybrid_search for
each, then deduplicates by chunk_id.

The original query is always included to ensure exact-match recall is preserved.
"""

import anthropic

from app.config import settings
from app.retrieval.search import hybrid_search

_CLIENT: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _CLIENT


def _generate_variants(query: str, n: int) -> list[str]:
    response = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Generate {n} semantically varied reformulations of this search query. "
                    "Each reformulation should approach the question from a different angle "
                    "or use different vocabulary, while preserving the original intent. "
                    "Return only the queries, one per line, no numbering or commentary.\n\n"
                    f"Query: {query}"
                ),
            }
        ],
    )
    lines = response.content[0].text.strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def multi_query_retrieve(
    query: str,
    user_id: str,
    document_id: str | None = None,
    top_k: int | None = None,
    n_variants: int | None = None,
) -> list[dict]:
    """
    Generate query variants, retrieve for each, return deduplicated chunks
    ranked by their best RRF score across all variants.

    Args:
        query:       original user query
        user_id:     passed through to hybrid_search for isolation
        document_id: optional document scope
        top_k:       chunks to fetch per variant (pre-rerank pool)
        n_variants:  number of extra query variants to generate

    Returns:
        Deduplicated list of chunk dicts, ordered by max RRF score descending
    """
    from app.config import settings as cfg

    n = n_variants or cfg.multi_query_variants
    k = top_k or cfg.retrieval_top_k

    variants = _generate_variants(query, n)
    all_queries = [query] + variants[:n]

    # Retrieve for every variant; track max rrf_score per chunk_id
    best: dict[str, dict] = {}
    for variant in all_queries:
        chunks = hybrid_search(variant, user_id, document_id, top_k=k)
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if cid not in best or chunk["rrf_score"] > best[cid]["rrf_score"]:
                best[cid] = chunk

    return sorted(best.values(), key=lambda c: c["rrf_score"], reverse=True)
