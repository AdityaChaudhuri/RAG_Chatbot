"""
Document type classifier.

Uses a HuggingFace zero-shot classification pipeline (BART-large-MNLI) to
assign each document one of five categories. The category is stored in
documents.doc_type and is used at query time to select the appropriate
prompt template and retrieval strategy.

Categories:
  legal      — contracts, court filings, legislation
  academic   — research papers, theses, journal articles
  financial  — reports, filings, invoices, statements
  technical  — API docs, RFCs, engineering specs
  general    — everything else
"""

from transformers import pipeline as hf_pipeline

_CLASSIFIER = None
_LABELS = ["legal", "academic", "financial", "technical", "general"]

# Only the first 2000 chars are needed for classification — beyond that the
# model's attention is diluted and inference slows without accuracy gains.
_SAMPLE_CHARS = 2_000


def _get_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU; set to 0 for GPU
        )
    return _CLASSIFIER


def classify_document(text: str) -> str:
    """
    Return the most likely document type label.

    Args:
        text: full document text

    Returns:
        One of: "legal", "academic", "financial", "technical", "general"
    """
    sample = text[:_SAMPLE_CHARS]
    result = _get_classifier()(sample, _LABELS, multi_label=False)
    return result["labels"][0]
