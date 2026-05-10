"""
Generation layer — powered by Gemini 2.0 Flash (Google AI free tier).

Gemini 2.0 Flash is fast, has a 1M token context window, and requires no
paid subscription on the free tier (rate limits apply).

Setup:
  1. Create a Gemini API key at https://aistudio.google.com/apikey
  2. Add GEMINI_API_KEY to your .env file
"""

from collections.abc import Generator

from google import genai
from google.genai import types

from app.config import settings
from app.generation.prompts import build_rag_prompt, build_summarise_prompt, system_prompt

_CLIENT: genai.Client | None = None

CHAT_MODEL = "gemini-2.0-flash"
_MAX_TOKENS_CHAT = 2048
_MAX_TOKENS_SUMMARY = 4096
_HISTORY_WINDOW = 10


def _get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=settings.gemini_api_key)
    return _CLIENT


def _convert_history(history: list[dict]) -> list[types.Content]:
    """Convert OpenAI-style history to Gemini Content objects (assistant → model)."""
    converted = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        if role in ("user", "model"):
            converted.append(
                types.Content(role=role, parts=[types.Part(text=msg["content"])])
            )
    return converted


def stream_answer(
    query: str,
    chunks: list[dict],
    history: list[dict],
    doc_type: str = "general",
) -> Generator[str, None, None]:
    """Stream Gemini's response token-by-token."""
    client = _get_client()
    prompt = build_rag_prompt(query, chunks, doc_type)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt(),
        max_output_tokens=_MAX_TOKENS_CHAT,
    )

    chat = client.chats.create(
        model=CHAT_MODEL,
        history=_convert_history(history[-_HISTORY_WINDOW:]),
        config=config,
    )

    for chunk in chat.send_message_stream(prompt):
        if chunk.text:
            yield chunk.text


def summarise(chunks: list[dict], doc_type: str = "general") -> str:
    """Generate a full-document summary (blocking)."""
    client = _get_client()
    prompt = build_summarise_prompt(chunks, doc_type)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt(),
            max_output_tokens=_MAX_TOKENS_SUMMARY,
        ),
    )
    return response.text
