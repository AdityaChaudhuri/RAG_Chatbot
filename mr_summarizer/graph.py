"""Knowledge graph building blocks: entity normalisation and co-occurrence edges.

Nodes are named entities found by spaCy. Two entities get an edge when they appear
in the same chunk; the edge weight counts how many chunks they share. This is a
simple, explainable graph: no LLM is needed to build it, so it is free and fast.
"""
import re
from collections import Counter
from itertools import combinations

# Entity types worth keeping for financial news
KEEP_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT"}

# Frequent spaCy false positives in news text
_STOP_ENTITIES = {"reuters", "thomson reuters", "u.s", "the", "inc", "corp", "ltd"}

_LEADING_THE = re.compile(r"^the\s+", re.I)
_POSSESSIVE = re.compile(r"(?:'s|’s)$")


def normalise(name: str) -> str:
    """Matching key for an entity: lowercase, no leading 'the', no possessive."""
    name = name.strip().strip(".,;:\"'()")
    name = _LEADING_THE.sub("", name)
    name = _POSSESSIVE.sub("", name)
    return re.sub(r"\s+", " ", name).lower()


def keep(key: str, label: str) -> bool:
    return label in KEEP_LABELS and len(key) > 1 and key not in _STOP_ENTITIES


def extract_mentions(docs) -> list[list[tuple[str, str, str]]]:
    """Turn spaCy docs into per-chunk lists of (key, display name, label),
    keeping only useful entity types and dropping noise."""
    out = []
    for doc in docs:
        seen = {}
        for ent in doc.ents:
            key = normalise(ent.text)
            if keep(key, ent.label_):
                seen.setdefault((key, ent.label_), ent.text.strip())
        out.append([(k, name, label) for (k, label), name in seen.items()])
    return out


def cooccurrence_edges(chunk_entities: list[set[int]]) -> Counter:
    """Count shared chunks for every entity pair. Pairs are stored as (low, high)."""
    edges: Counter = Counter()
    for ids in chunk_entities:
        for a, b in combinations(sorted(ids), 2):
            edges[(a, b)] += 1
    return edges


def entities_in_query(query: str, known: dict[str, list[int]]) -> list[int]:
    """Find known entity keys mentioned in a free-text question.

    spaCy NER is unreliable on short lowercase questions, so instead we look for
    each known entity key as a whole-word phrase inside the normalised query.
    Longer names win, so 'goldman sachs' is matched before 'goldman'.
    """
    q = " " + re.sub(r"[^\w\s&.-]", " ", query.lower()) + " "
    q = re.sub(r"\s+", " ", q)
    found: list[int] = []
    for key in sorted(known, key=len, reverse=True):
        needle = f" {key} "
        if needle in q:
            found.extend(known[key])  # one key can be several nodes (e.g. ORG and PRODUCT)
            q = q.replace(needle, " ")  # stop shorter keys matching inside it
    return found
