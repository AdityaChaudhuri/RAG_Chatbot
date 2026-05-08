"""
Multi-query retrieval — uses Gemini 2.0 Flash to generate query variants.

Rephrasing is a simple task that Gemini Flash handles quickly, so we use the
same model as the main generation layer rather than spinning up a second one.
"""

import google.generativeai as genai

from app.config import settings
from app.retrieval.search import hybrid_search

_MODEL: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _MODEL
    if _MODEL is None:
        genai.configure(api_key=settings.gemini_api_key)
        _MODEL = genai.GenerativeModel("gemini-2.0-flash")
    return _MODEL


def _generate_variants(query: str, n: int) -> list[str]:
    response = _get_model().generate_content(
        (
            f"Generate {n} semantically varied reformulations of this search query. "
            "Each reformulation should approach the question from a different angle "
            "or use different vocabulary, while preserving the original intent. "
            "Return only the queries, one per line, no numbering or commentary.\n\n"
            f"Query: {query}"
        ),
        generation_config=genai.GenerationConfig(max_output_tokens=300),
    )
    lines = response.text.strip().splitlines()
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
    """
    from app.config import settings as cfg

    n = n_variants or cfg.multi_query_variants
    k = top_k or cfg.retrieval_top_k

    variants = _generate_variants(query, n)
    all_queries = [query] + variants[:n]

    best: dict[str, dict] = {}
    for variant in all_queries:
        chunks = hybrid_search(variant, user_id, document_id, top_k=k)
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if cid not in best or chunk["rrf_score"] > best[cid]["rrf_score"]:
                best[cid] = chunk

    return sorted(best.values(), key=lambda c: c["rrf_score"], reverse=True)
