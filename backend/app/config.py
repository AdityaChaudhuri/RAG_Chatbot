from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str

    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    database_url: str

    # Chunking thresholds
    chunk_similarity_threshold: float = 0.5
    chunk_min_tokens: int = 100
    chunk_max_tokens: int = 512

    # Retrieval defaults
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    multi_query_variants: int = 4

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")


settings = Settings()  # type: ignore[call-arg]
