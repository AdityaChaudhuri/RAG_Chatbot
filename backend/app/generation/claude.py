"""
Claude generation layer.

Two modes:
  stream_answer   — SSE token stream for the chat UI (RAG Q&A)
  summarise       — blocking full-document summary
"""

from collections.abc import Generator

import anthropic

from app.config import settings
from app.generation.prompts import build_rag_prompt, build_summarise_prompt, system_prompt

_CLIENT: anthropic.Anthropic | None = None
_CHAT_MODEL = "claude-sonnet-4-6"
_FAST_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS_CHAT = 2048
_MAX_TOKENS_SUMMARY = 4096
_HISTORY_WINDOW = 10  # number of prior turns to include


def _get_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _CLIENT


def _build_messages(
    query: str,
    chunks: list[dict],
    history: list[dict],
    doc_type: str,
) -> list[dict]:
    """Assemble the messages array: prior turns + current RAG prompt."""
    prior = history[-_HISTORY_WINDOW:]
    current_prompt = build_rag_prompt(query, chunks, doc_type)
    return [*prior, {"role": "user", "content": current_prompt}]


def stream_answer(
    query: str,
    chunks: list[dict],
    history: list[dict],
    doc_type: str = "general",
) -> Generator[str, None, None]:
    """
    Stream Claude's response token-by-token.

    Args:
        query:    user question
        chunks:   compressed, re-ranked context chunks
        history:  list of {role, content} dicts from chat_history
        doc_type: used to select the prompt template

    Yields:
        str tokens as they arrive from the API
    """
    messages = _build_messages(query, chunks, history, doc_type)

    with _get_client().messages.stream(
        model=_CHAT_MODEL,
        max_tokens=_MAX_TOKENS_CHAT,
        system=system_prompt(),
        messages=messages,
    ) as stream:
        for token in stream.text_stream:
            yield token


def summarise(chunks: list[dict], doc_type: str = "general") -> str:
    """
    Generate a structured full-document summary (blocking).

    Args:
        chunks:   all chunks for the document, ordered by chunk_index
        doc_type: document type for prompt routing

    Returns:
        Markdown-formatted summary string
    """
    prompt = build_summarise_prompt(chunks, doc_type)
    response = _get_client().messages.create(
        model=_CHAT_MODEL,
        max_tokens=_MAX_TOKENS_SUMMARY,
        system=system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
