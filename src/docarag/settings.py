from typing import Any, Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Model providers are selected per component (LLM, embeddings, reranker)
    through environment variables only; no code change is needed to switch
    between OpenRouter, a local OpenAI-compatible server or the gRPC reranker.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="forbid"
    )

    # --- OpenRouter -----------------------------------------------------------
    # One key for every component whose base_url points at OpenRouter; a
    # component-specific *_API_KEY still wins when set
    openrouter_api_key: SecretStr = SecretStr("")

    # --- LLM (chat) ---------------------------------------------------------
    llm_provider: Literal["openai", "anthropic"] = "openai"
    # Any OpenAI-compatible chat endpoint: OpenRouter, llama.cpp, vLLM, Ollama
    llm_base_url: str | None = "https://openrouter.ai/api/v1"
    llm_api_key: SecretStr = SecretStr("EMPTY")
    llm_model: str = "qwen/qwen3.8-flash"
    llm_temperature: float = 0.7
    llm_timeout: int = 120
    llm_max_retries: int = 2
    # Extra JSON merged into the request body, e.g. {"reasoning": {"enabled": false}}
    llm_extra_body: dict[str, Any] | None = None
    # Extra HTTP headers, e.g. {"HTTP-Referer": "...", "X-Title": "doc-arag"}
    llm_default_headers: dict[str, str] | None = None

    # --- LLM: legacy Anthropic branch, used only when llm_provider=anthropic --
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None
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

    # --- Embeddings (OpenAI-compatible /embeddings) -------------------------
    # OpenRouter (qwen/qwen3-embedding-8b, 4096 dims) or a local llama.cpp /
    # Ollama server; see compose.models.yml for the local stack
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    embedding_api_key: SecretStr = SecretStr("EMPTY")
    embedding_model: str = "qwen/qwen3-embedding-8b"
    # Sent only when set; some models accept a reduced output dimension
    embedding_dimensions: int | None = None
    # Texts per HTTP request (DashScope caps this at 10, local servers accept more)
    embedding_batch_size: int = 16
    embedding_timeout: int = 120
    embedding_max_retries: int = 3

    # --- Reranker -------------------------------------------------------------
    reranker_provider: Literal["openai-rerank", "grpc", "none"] = "openai-rerank"
    # openai-rerank: POST {base_url}/rerank with {model, query, documents, top_n}
    # -> results[{index, relevance_score}] (OpenRouter, llama.cpp, vLLM, Jina, Cohere)
    reranker_base_url: str | None = "https://openrouter.ai/api/v1"
    reranker_api_key: SecretStr = SecretStr("EMPTY")
    reranker_model: str = "qwen/qwen3-reranker-8b"
    # grpc: the rag-services reranker (compose service name must be reranker-service)
    reranker_service_url: str = "reranker-service:8352"
    reranker_timeout: int = 30
    reranker_max_retries: int = 1

    # --- Auth (see src/docarag/auth.py) ----------------------------------------
    # The API never checks passwords. When auth_trusted_headers is on, the
    # identity comes from Remote-User / Remote-Groups set by the edge proxy
    # (Caddy forward_auth -> Authelia) and every request must also carry
    # X-Auth-Proxy-Secret = auth_proxy_secret: containers on the shared docker
    # network can reach the API directly, so the headers alone prove nothing.
    # Off (default): every caller is an anonymous administrator (local, tests).
    auth_trusted_headers: bool = False
    auth_proxy_secret: SecretStr = SecretStr("")
    # Group whose members may upload, embed, list and delete documents; the
    # Authelia rules in doc-arag-client are templated from the same variable
    auth_admin_group: str = "admins"

    # --- Storage --------------------------------------------------------------
    minio_endpoint: str
    minio_access_key: SecretStr
    minio_secret_key: SecretStr
    minio_bucket: str
    minio_secure: bool = True

    weaviate_host: str = "weaviate"
    weaviate_port: int = 8080
    weaviate_insert_batch_size: int = 100
    # Compare the live embedding dimension with stored vectors at startup
    startup_verify_embedding_dimension: bool = True

    # --- Chunking -------------------------------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Markdown is split by headers first; a section is kept whole up to this
    # many characters (API embedders accept thousands of tokens, so the old
    # 512-token ceiling no longer applies)
    md_chunk_size: int = 1800
    md_chunk_overlap: int = 200
    max_file_size_mb: int = 50

    # --- Retrieval / agent ------------------------------------------------------
    initial_retrieval_k: int = 20
    rerank_top_k: int = 5
    agent_confidence_threshold: float = 0.7

    @staticmethod
    def _is_openrouter(base_url: str | None) -> bool:
        return bool(base_url) and str(base_url).startswith("https://openrouter.ai")

    @model_validator(mode="after")
    def _apply_openrouter_key(self) -> "Settings":
        """Fill component keys from OPENROUTER_API_KEY where the base_url is OpenRouter."""
        shared = self.openrouter_api_key.get_secret_value()
        if not shared:
            return self
        for key_field, url in (
            ("llm_api_key", self.llm_base_url),
            ("embedding_api_key", self.embedding_base_url),
            ("reranker_api_key", self.reranker_base_url),
        ):
            current: SecretStr = getattr(self, key_field)
            if self._is_openrouter(url) and current.get_secret_value() in ("", "EMPTY"):
                setattr(self, key_field, SecretStr(shared))
        return self

    @model_validator(mode="after")
    def _validate_providers(self) -> "Settings":
        if self.llm_provider == "anthropic" and not (
            self.anthropic_api_key and self.anthropic_model
        ):
            raise ValueError(
                "llm_provider=anthropic requires ANTHROPIC_API_KEY and ANTHROPIC_MODEL"
            )
        if self.llm_provider == "openai" and not self.llm_model:
            raise ValueError("llm_provider=openai requires LLM_MODEL")
        if self.reranker_provider == "openai-rerank" and not self.reranker_base_url:
            raise ValueError(
                "reranker_provider=openai-rerank requires RERANKER_BASE_URL"
            )
        if self.auth_trusted_headers and not self.auth_proxy_secret.get_secret_value():
            raise ValueError("AUTH_TRUSTED_HEADERS=true requires AUTH_PROXY_SECRET")
        return self


# Required fields are supplied by the environment, not by the call site
settings = Settings()  # type: ignore[call-arg]
