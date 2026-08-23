from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="forbid"
    )

    anthropic_api_key: str
    anthropic_model: str
    anthropic_proxy_url: str | None = None
    anthropic_proxy_user: str | None = None
    anthropic_proxy_pass: str | None = None

    @property
    def anthropic_proxy(self) -> str | None:
        """Build full proxy URL with credentials if provided."""
        if not self.anthropic_proxy_url:
            return None

        if self.anthropic_proxy_user and self.anthropic_proxy_pass:
            # Parse URL and insert credentials: http://user:pass@host:port
            if "://" in self.anthropic_proxy_url:
                scheme, rest = self.anthropic_proxy_url.split("://", 1)
                return f"{scheme}://{self.anthropic_proxy_user}:{self.anthropic_proxy_pass}@{rest}"

        return self.anthropic_proxy_url

    minio_endpoint: str
    minio_access_key: SecretStr
    minio_secret_key: SecretStr
    minio_bucket: str
    minio_secure: bool = True

    weaviate_host: str = "weaviate"
    weaviate_port: int = 8080
    weaviate_collection: str = "Documents"

    chunk_size: int = 512
    chunk_overlap: int = 64
    # Markdown is split by headers, so a section needs a larger budget than a
    # PDF page slice; 900 characters stay under the 512-token embedding limit
    md_chunk_size: int = 900
    md_chunk_overlap: int = 100
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

    agent_confidence_threshold: float = 0.7
    anthropic_temperature: float = 0.7
    reranker_service_url: str = "reranker-service:8352"
    reranker_timeout: int = 30


# Required fields are supplied by the environment, not by the call site
settings = Settings()  # type: ignore[call-arg]
