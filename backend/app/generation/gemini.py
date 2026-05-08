"""
Generation layer — powered by Gemini 2.0 Flash (Google AI free tier).

Gemini 2.0 Flash is fast, has a 1M token context window, and requires no
paid subscription on the free tier (rate limits apply).

Setup:
  1. Create a Gemini API key at https://aistudio.google.com/apikey
  2. Add GEMINI_API_KEY to your .env file
"""

from collections.abc import Generator

import google.generativeai as genai

from app.config import settings
from app.generation.prompts import build_rag_prompt, build_summarise_prompt, system_prompt

_MODEL: genai.GenerativeModel | None = None

CHAT_MODEL = "gemini-2.0-flash"
_MAX_TOKENS_CHAT = 2048
_MAX_TOKENS_SUMMARY = 4096
_HISTORY_WINDOW = 10


def _get_model() -> genai.GenerativeModel:
    global _MODEL
    if _MODEL is None:
        genai.configure(api_key=settings.gemini_api_key)
        _MODEL = genai.GenerativeModel(
            CHAT_MODEL,
            system_instruction=system_prompt(),
        )
    return _MODEL


def _convert_history(history: list[dict]) -> list[dict]:
    """Convert OpenAI-style history to Gemini format (assistant → model)."""
    converted = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        if role in ("user", "model"):
            converted.append({"role": role, "parts": [{"text": msg["content"]}]})
    return converted


def stream_answer(
    query: str,
    chunks: list[dict],
    history: list[dict],
    doc_type: str = "general",
) -> Generator[str, None, None]:
    """Stream Gemini's response token-by-token."""
    model = _get_model()
    prompt = build_rag_prompt(query, chunks, doc_type)
    gemini_history = _convert_history(history[-_HISTORY_WINDOW:])

    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(
        prompt,
        stream=True,
        generation_config=genai.GenerationConfig(max_output_tokens=_MAX_TOKENS_CHAT),
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text


def summarise(chunks: list[dict], doc_type: str = "general") -> str:
    """Generate a full-document summary (blocking)."""
    model = _get_model()
    prompt = build_summarise_prompt(chunks, doc_type)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=_MAX_TOKENS_SUMMARY),
    )
    return response.text
