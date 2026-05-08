"""
Contextual compression — uses Gemini 2.0 Flash.

Strips irrelevant sentences from each retrieved chunk before passing context
to the main generation step. Keeps the token budget tight.
"""

import google.generativeai as genai

from app.config import settings

_MODEL: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _MODEL
    if _MODEL is None:
        genai.configure(api_key=settings.gemini_api_key)
        _MODEL = genai.GenerativeModel("gemini-2.0-flash")
    return _MODEL


def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Strip irrelevant sentences from each chunk. Drops chunks that compress to nothing."""
    compressed: list[dict] = []

    for chunk in chunks:
        response = _get_model().generate_content(
            (
                f"Question: {query}\n\n"
                f"Document excerpt:\n{chunk['content']}\n\n"
                "Extract only the sentences from the excerpt that directly help "
                "answer the question. Preserve exact wording. "
                "If nothing is relevant, reply with an empty response."
            ),
            generation_config=genai.GenerationConfig(max_output_tokens=512),
        )
        text = response.text.strip()
        if text:
            compressed.append({**chunk, "content": text})

    return compressed
