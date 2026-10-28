"""Lazy-loaded local models. Each loads once on first use (they take a few seconds)."""
from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


def embed_documents(texts: list[str], batch_size: int = 64):
    # normalised vectors: cosine distance in pgvector then equals 1 - dot product
    return embedder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def embed_query(query: str):
    return embedder().encode(settings.embed_query_prefix + query, normalize_embeddings=True)


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    """Cross-encoder reads query and passage together, so it judges relevance far
    better than comparing two separately computed embeddings. It is too slow to run
    over the whole corpus, which is why it only re-orders the fused shortlist."""
    if not passages:
        return []
    return [float(s) for s in reranker().predict([(query, p) for p in passages])]


@lru_cache(maxsize=1)
def ner():
    import spacy

    # only the NER component is needed; disabling the rest makes it much faster
    return spacy.load(settings.spacy_model, disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
