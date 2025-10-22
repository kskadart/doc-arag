from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="forbid"
    )

    # Anthropic API settings
    anthropic_api_key: str
    anthropic_model: str

    # MinIO/S3 settings
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # Weaviate settings
    weaviate_host: str = "weaviate"
    weaviate_port: int = 8080
    weaviate_collection: str = "Documents"

    # Model settings
    embedding_model_name: str = "google/embeddinggemma-300m"  # EmbeddingGemma 300M
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Processing settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 50

    # Retrieval settings
    initial_retrieval_k: int = 20
    rerank_top_k: int = 5

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8103

    # Embedding service settings
    embedding_service_url: str = "embedding-service:8351"
    embedding_service_timeout: int = 30
    embedding_use_async: bool = True


# Global settings instance
settings = Settings()
