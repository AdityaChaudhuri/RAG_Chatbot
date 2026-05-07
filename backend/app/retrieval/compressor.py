"""
Contextual compression.

After re-ranking, chunks may still contain sentences irrelevant to the
specific question. This module uses Claude Haiku (fast, cheap) to strip
those sentences — reducing prompt token count and noise before the final
generation call.

If compression produces an empty string (chunk is entirely irrelevant),
the chunk is dropped from the final context.
"""

import anthropic

from app.config import settings

_CLIENT: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _CLIENT


def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Strip irrelevant sentences from each chunk.

    Args:
        query:  the user's question
        chunks: top-K chunks from reranker.rerank()

    Returns:
        Filtered list with compressed content. Chunks that compress to
        nothing are excluded entirely.
    """
    client = _get_client()
    compressed: list[dict] = []

    for chunk in chunks:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Document excerpt:\n{chunk['content']}\n\n"
                        "Extract only the sentences from the excerpt that directly help "
                        "answer the question. Preserve exact wording. "
                        "If nothing is relevant, reply with an empty response."
                    ),
                }
            ],
        )
        text = response.content[0].text.strip()
        if text:
            compressed.append({**chunk, "content": text})

    return compressed
