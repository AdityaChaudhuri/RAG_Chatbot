"""All settings in one place, read from environment variables (.env file)."""
import os
from dataclasses import dataclass

try:  # python-dotenv is optional; it just loads .env into os.environ
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://mr:mr@localhost:5432/mr_summarizer"
    )

    # Embedding model: small, fast on CPU, 384 dimensions (matches sql/schema.sql)
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    # bge models are trained to expect this prefix on queries (not on documents)
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "

    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")

    # Retrieval knobs
    candidates_per_retriever: int = 30   # how many results each retriever returns
    rrf_k: int = 60                      # standard RRF constant from Cormack et al. (2009)
    rerank_top_n: int = 20               # fused candidates passed to the cross-encoder
    final_top_k: int = 6                 # chunks given to the LLM

    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")


settings = Settings()
