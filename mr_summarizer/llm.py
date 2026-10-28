"""Prompt construction and streaming answers from Claude or a local Ollama model."""
import json
from typing import Iterator

from .config import settings

SYSTEM_PROMPT = """You answer questions about financial news using only the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not add outside knowledge.
- Cite every factual sentence with the source number in square brackets, like [2] or [1][3].
- If the sources do not contain the answer, say so plainly and say what they do cover.
- Keep the answer short: a direct answer first, then supporting detail. No preamble."""


def build_user_prompt(question: str, sources: list[dict]) -> str:
    """sources: dicts with 'title' and 'content', in the order they will be numbered."""
    blocks = [
        f"[{i}] {s['title']}\n{s['content']}" for i, s in enumerate(sources, start=1)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def stream_answer(question: str, sources: list[dict]) -> Iterator[str]:
    prompt = build_user_prompt(question, sources)
    if settings.llm_provider == "ollama":
        yield from _stream_ollama(prompt)
    else:
        yield from _stream_anthropic(prompt)


def _stream_anthropic(prompt: str) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    with client.messages.stream(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        yield from stream.text_stream


def _stream_ollama(prompt: str) -> Iterator[str]:
    import requests

    resp = requests.post(
        f"{settings.ollama_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("done"):
            break
        yield msg.get("message", {}).get("content", "")
