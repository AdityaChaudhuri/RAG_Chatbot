"""Cleaning and chunking. Pure Python, no heavy dependencies, fully unit-tested."""
import re

# Boilerplate that scraped Reuters pages carry around the real article text
_BOILERPLATE = [
    re.compile(r"Updated \d+ (?:minutes?|hours?|days?) ago", re.I),
    re.compile(r"\b\d+ Min Read\b", re.I),
    re.compile(r"Reuters Staff", re.I),
    re.compile(r"Reporting by [^.]*\.?$", re.I | re.M),
    re.compile(r"Editing by [^.]*\.?$", re.I | re.M),
    re.compile(r"Our Standards:.*$", re.I | re.M),
]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def clean(text: str) -> str:
    """Strip page boilerplate and normalise whitespace."""
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk(text: str, max_words: int = 180, overlap_words: int = 40) -> list[str]:
    """Split text into chunks of whole sentences, about max_words long.

    Consecutive chunks share roughly overlap_words of trailing sentences so a fact
    that straddles a boundary still appears intact in at least one chunk.
    A single sentence longer than max_words becomes its own chunk.
    """
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        n = len(sentence.split())
        if current and current_words + n > max_words:
            chunks.append(" ".join(current))
            # carry trailing sentences into the next chunk as overlap
            carry: list[str] = []
            carry_words = 0
            for s in reversed(current):
                w = len(s.split())
                if carry_words + w > overlap_words:
                    break
                carry.insert(0, s)
                carry_words += w
            current, current_words = carry, carry_words
        current.append(sentence)
        current_words += n

    if current:
        last = " ".join(current)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks
