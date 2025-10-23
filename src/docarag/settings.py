from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="forbid"
    )

    anthropic_api_key: str
    anthropic_model: str

    minio_endpoint: str
    minio_access_key: SecretStr
    minio_secret_key: SecretStr
    minio_bucket: str
    minio_secure: bool = True

    weaviate_host: str = "weaviate"
    weaviate_port: int = 8080
    weaviate_collection: str = "Documents"

    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 50

    initial_retrieval_k: int = 20
    rerank_top_k: int = 5

    api_host: str = "0.0.0.0"
    api_port: int = 8103

    embedding_service_url: str = "embedding-service:8351"
    embedding_service_timeout: int = 300  # Increased to 5 minutes for large batches
    embedding_use_async: bool = True
    embedding_max_length: int = 512
    embedding_pooling_strategy: str = "mean"
    embedding_normalize: bool = True
    embedding_batch_size: int = 32  # Process in smaller batches


settings = Settings()
