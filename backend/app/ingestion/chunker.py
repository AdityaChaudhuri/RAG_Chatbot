"""
Semantic chunker — splits documents at topic boundaries rather than fixed character counts.

A boundary is any point where cosine similarity between two adjacent sentences
drops below the threshold, indicating a topic shift. Chunks that come out too
small are merged into their successor; chunks that come out too large are re-split
at sentence boundaries.
"""

import re

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from app.config import settings

_ENCODER = tiktoken.get_encoding("cl100k_base")
_EMBED_MODEL: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        # Lightweight model for boundary detection only — not used for retrieval.
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def semantic_chunk(
    pages: list[dict],
    similarity_threshold: float | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[dict]:
    """
    Args:
        pages: output of parser.parse_pdf()
        similarity_threshold: split when adjacent-sentence similarity drops below this
        min_tokens: merge chunks smaller than this into the next chunk
        max_tokens: hard cap — oversized chunks are re-split at sentence boundaries

    Returns:
        List of dicts: {content, page_nums, token_count, chunk_index}
    """
    threshold = similarity_threshold or settings.chunk_similarity_threshold
    min_tok = min_tokens or settings.chunk_min_tokens
    max_tok = max_tokens or settings.chunk_max_tokens

    sentences: list[dict] = []
    for page in pages:
        for sent in _split_sentences(page["content"]):
            sentences.append({"text": sent, "page_num": page["page_num"]})

    if not sentences:
        return []

    texts = [s["text"] for s in sentences]
    embeddings: np.ndarray = _get_embed_model().encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )

    similarities = [
        _cosine(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Seed with start and end; add interior split points where similarity drops.
    boundaries = {0}
    for i, sim in enumerate(similarities):
        if sim < threshold:
            boundaries.add(i + 1)
    boundaries.add(len(sentences))
    sorted_boundaries = sorted(boundaries)

    raw: list[dict] = []
    for i in range(len(sorted_boundaries) - 1):
        start = sorted_boundaries[i]
        end = sorted_boundaries[i + 1]
        span = sentences[start:end]
        content = " ".join(s["text"] for s in span)
        page_nums = sorted({s["page_num"] for s in span})
        raw.append({
            "content": content,
            "page_nums": page_nums,
            "token_count": _token_count(content),
        })

    # Merge undersized chunks into their successor.
    merged: list[dict] = []
    buffer: dict | None = None
    for chunk in raw:
        if buffer is None:
            buffer = chunk
        elif buffer["token_count"] < min_tok:
            buffer["content"] += " " + chunk["content"]
            buffer["page_nums"] = sorted(set(buffer["page_nums"] + chunk["page_nums"]))
            buffer["token_count"] = _token_count(buffer["content"])
        else:
            merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)

    # Hard-split any chunk that still exceeds max_tokens at sentence boundaries.
    final: list[dict] = []
    for chunk in merged:
        if chunk["token_count"] <= max_tok:
            final.append(chunk)
        else:
            sub_sentences = _split_sentences(chunk["content"])
            sub_buf = []
            for sent in sub_sentences:
                sub_buf.append(sent)
                if _token_count(" ".join(sub_buf)) >= max_tok:
                    content = " ".join(sub_buf[:-1])
                    if content.strip():
                        final.append({
                            "content": content,
                            "page_nums": chunk["page_nums"],
                            "token_count": _token_count(content),
                        })
                    sub_buf = [sent]
            if sub_buf:
                content = " ".join(sub_buf)
                final.append({
                    "content": content,
                    "page_nums": chunk["page_nums"],
                    "token_count": _token_count(content),
                })

    for idx, chunk in enumerate(final):
        chunk["chunk_index"] = idx

    return final
