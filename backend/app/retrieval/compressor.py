"""
Contextual compression — uses Gemini 2.0 Flash.

Strips irrelevant sentences from each retrieved chunk before passing context
to the main generation step. Keeps the token budget tight.
"""

from google import genai
from google.genai import types

from app.config import settings

_CLIENT: genai.Client | None = None


def _get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=settings.gemini_api_key)
    return _CLIENT


def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Strip irrelevant sentences from each chunk. Drops chunks that compress to nothing."""
    client = _get_client()
    compressed: list[dict] = []

    for chunk in chunks:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                f"Question: {query}\n\n"
                f"Document excerpt:\n{chunk['content']}\n\n"
                "Extract only the sentences from the excerpt that directly help "
                "answer the question. Preserve exact wording. "
                "If nothing is relevant, reply with an empty response."
            ),
            config=types.GenerateContentConfig(max_output_tokens=512),
        )
        text = response.text.strip()
        if text:
            compressed.append({**chunk, "content": text})

    return compressed
