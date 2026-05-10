"""
Named Entity Recognition using spaCy en_core_web_lg.

Extracts structured entity metadata from the full document text and stores
it as JSONB in the documents table. This enables SQL queries like:

    SELECT * FROM documents WHERE entity_tags @> '{"orgs": ["OpenAI"]}';

Run after install:  python -m spacy download en_core_web_lg
"""

from collections import defaultdict

import spacy

_NLP: spacy.language.Language | None = None

# spaCy label → our JSONB key mapping
_LABEL_MAP = {
    "PERSON": "people",
    "ORG": "orgs",
    "DATE": "dates",
    "TIME": "times",
    "GPE": "locations",
    "LOC": "locations",
    "MONEY": "amounts",
    "LAW": "laws",
}

# spaCy struggles on very large texts — cap at 100k chars for NER
_MAX_CHARS = 100_000


def _get_nlp() -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_lg")
    return _NLP


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Run NER on document text and return a deduplicated entity dict.

    Args:
        text: full document text (output of parser.full_text)

    Returns:
        {
          "people":    ["John Smith", ...],
          "orgs":      ["Anthropic", ...],
          "dates":     ["January 2024", ...],
          "locations": ["San Francisco", ...],
          ...
        }
    """
    nlp = _get_nlp()
    doc = nlp(text[:_MAX_CHARS])

    entities: dict[str, set[str]] = defaultdict(set)
    for ent in doc.ents:
        key = _LABEL_MAP.get(ent.label_)
        if key:
            cleaned = ent.text.strip()
            if len(cleaned) > 1:  # skip single-char noise
                entities[key].add(cleaned)

    return {k: sorted(v) for k, v in entities.items()}
