"""
Semantic chunker.

Splits a document into chunks by detecting topic boundaries — points where
the cosine similarity between adjacent sentences drops below a threshold.
This preserves semantic coherence within each chunk, unlike fixed-size
splitting which can sever a thought mid-sentence.

Algorithm:
  1. Sentence-tokenise all pages.
  2. Embed every sentence with a lightweight bi-encoder.
  3. Compute cosine similarity between each consecutive sentence pair.
  4. Split where similarity < threshold (topic boundary).
  5. Merge chunks that are below min_tokens.
  6. Return chunks with page provenance and token counts.
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
        # Small, fast model used only for boundary detection — not for retrieval.
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _split_sentences(text: str) -> list[str]:
    # Split on sentence-ending punctuation followed by whitespace.
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
        similarity_threshold: split when adjacent-sentence similarity < this
        min_tokens: merge chunks smaller than this into the next chunk
        max_tokens: hard cap — oversized chunks are split at sentence boundaries

    Returns:
        List of dicts: {content, page_nums, token_count, chunk_index}
    """
    threshold = similarity_threshold or settings.chunk_similarity_threshold
    min_tok = min_tokens or settings.chunk_min_tokens
    max_tok = max_tokens or settings.chunk_max_tokens

    # ---- 1. Flatten sentences with page provenance -------------------------
    sentences: list[dict] = []
    for page in pages:
        for sent in _split_sentences(page["content"]):
            sentences.append({"text": sent, "page_num": page["page_num"]})

    if not sentences:
        return []

    # ---- 2. Embed all sentences in one batched call -----------------------
    texts = [s["text"] for s in sentences]
    model = _get_embed_model()
    embeddings: np.ndarray = model.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )

    # ---- 3. Adjacent cosine similarities ----------------------------------
    similarities = [
        _cosine(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # ---- 4. Boundary detection --------------------------------------------
    boundaries = {0}
    for i, sim in enumerate(similarities):
        if sim < threshold:
            boundaries.add(i + 1)
    boundaries.add(len(sentences))
    sorted_boundaries = sorted(boundaries)

    # ---- 5. Build raw chunks ---------------------------------------------
    raw: list[dict] = []
    for i in range(len(sorted_boundaries) - 1):
        start = sorted_boundaries[i]
        end = sorted_boundaries[i + 1]
        span = sentences[start:end]
        content = " ".join(s["text"] for s in span)
        page_nums = sorted({s["page_num"] for s in span})
        raw.append(
            {
                "content": content,
                "page_nums": page_nums,
                "token_count": _token_count(content),
            }
        )

    # ---- 6. Merge undersized chunks into successor -----------------------
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

    # ---- 7. Hard-split chunks that exceed max_tokens --------------------
    final: list[dict] = []
    for chunk in merged:
        if chunk["token_count"] <= max_tok:
            final.append(chunk)
        else:
            # Split at sentence boundaries until under max_tok
            sub_sentences = _split_sentences(chunk["content"])
            sub_buf = []
            for sent in sub_sentences:
                sub_buf.append(sent)
                if _token_count(" ".join(sub_buf)) >= max_tok:
                    content = " ".join(sub_buf[:-1])
                    if content.strip():
                        final.append(
                            {
                                "content": content,
                                "page_nums": chunk["page_nums"],
                                "token_count": _token_count(content),
                            }
                        )
                    sub_buf = [sent]
            if sub_buf:
                content = " ".join(sub_buf)
                final.append(
                    {
                        "content": content,
                        "page_nums": chunk["page_nums"],
                        "token_count": _token_count(content),
                    }
                )

    # ---- 8. Attach chunk index ------------------------------------------
    for idx, chunk in enumerate(final):
        chunk["chunk_index"] = idx

    return final
