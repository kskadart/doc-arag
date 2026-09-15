"""Tests for provider-related settings validation."""

import json

import pytest
from pydantic import ValidationError

from src.docarag.settings import Settings

_MINIO = {
    "minio_endpoint": "localhost:9000",
    "minio_access_key": "k",
    "minio_secret_key": "s",
    "minio_bucket": "b",
}


def test_defaults_point_at_openrouter_and_grpc_reranker(monkeypatch):
    for key in (
        "EMBEDDING_API_KEY",
        "RERANKER_BASE_URL",
        "RERANKER_API_KEY",
        "LLM_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    for key in (
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "RERANKER_PROVIDER",
        "RERANKER_SERVICE_URL",
        "STARTUP_VERIFY_EMBEDDING_DIMENSION",
    ):
        monkeypatch.delenv(key, raising=False)

    s = Settings(_env_file=None, **_MINIO)

    assert s.llm_provider == "openai"
    assert s.llm_base_url == "https://openrouter.ai/api/v1"
    assert s.llm_model == "qwen/qwen3.8-flash"
    assert s.embedding_base_url == "https://openrouter.ai/api/v1"
    assert s.embedding_model == "qwen/qwen3-embedding-8b"
    assert s.reranker_provider == "openai-rerank"
    assert s.reranker_base_url == "https://openrouter.ai/api/v1"
    assert s.reranker_model == "qwen/qwen3-reranker-8b"
    assert s.reranker_service_url == "reranker-service:8352"
    assert s.startup_verify_embedding_dimension is True
    assert s.md_chunk_size == 1800
    assert not hasattr(s, "weaviate_collection")
    assert not hasattr(s, "embedding_service_url")


def test_anthropic_provider_requires_key_and_model():
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings(_env_file=None, llm_provider="anthropic", **_MINIO)


def test_http_reranker_requires_base_url(monkeypatch):
    monkeypatch.delenv("RERANKER_BASE_URL", raising=False)
    with pytest.raises(ValidationError, match="RERANKER_BASE_URL"):
        Settings(
            _env_file=None,
            reranker_provider="openai-rerank",
            reranker_base_url=None,
            **_MINIO,
        )


def test_llm_extra_body_parses_json_from_env(monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", json.dumps({"reasoning": {"enabled": False}}))

    s = Settings(_env_file=None, **_MINIO)

    assert s.llm_extra_body == {"reasoning": {"enabled": False}}


def test_unknown_env_key_in_dotenv_is_rejected(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MINIO_ENDPOINT=x\nMINIO_ACCESS_KEY=k\nMINIO_SECRET_KEY=s\nMINIO_BUCKET=b\nWEAVIATE_COLLECTION=Old\n"
    )

    with pytest.raises(ValidationError, match="extra"):
        Settings(_env_file=env_file)


def test_openrouter_key_fills_components_pointing_at_openrouter(monkeypatch):
    for key in (
        "LLM_API_KEY",
        "EMBEDDING_API_KEY",
        "RERANKER_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    s = Settings(
        _env_file=None,
        openrouter_api_key="sk-or-shared",
        llm_base_url="https://openrouter.ai/api/v1",
        embedding_base_url="http://llama-embed:8080/v1",
        embedding_api_key="local-key",
        reranker_base_url="https://openrouter.ai/api/v1",
        **_MINIO,
    )

    assert s.llm_api_key.get_secret_value() == "sk-or-shared"
    assert s.reranker_api_key.get_secret_value() == "sk-or-shared"
    assert s.embedding_api_key.get_secret_value() == "local-key"


def test_component_key_wins_over_openrouter_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    s = Settings(
        _env_file=None,
        openrouter_api_key="sk-or-shared",
        llm_api_key="sk-mine",
        **_MINIO,
    )
    assert s.llm_api_key.get_secret_value() == "sk-mine"


def test_session_defaults(monkeypatch):
    for key in (
        "SESSION_STORE",
        "SESSION_TTL_DAYS",
        "SESSION_HISTORY_MESSAGES",
        "SESSION_SUMMARY_AFTER_MESSAGES",
        "SESSION_MESSAGE_MAX_CHARS",
        "SESSION_MAX_STORED_MESSAGES",
        "SESSION_CLEANUP_INTERVAL_MINUTES",
    ):
        monkeypatch.delenv(key, raising=False)

    s = Settings(_env_file=None, **_MINIO)

    assert s.session_store == "weaviate"
    assert s.session_ttl_days == 7
    assert s.session_history_messages == 6
    assert s.session_summary_after_messages == 12
    assert s.session_message_max_chars == 1200
    assert s.session_max_stored_messages == 200
    assert s.session_cleanup_interval_minutes == 60


def test_session_summary_threshold_must_exceed_history_window():
    with pytest.raises(ValidationError, match="SESSION_SUMMARY_AFTER_MESSAGES"):
        Settings(
            _env_file=None,
            session_history_messages=6,
            session_summary_after_messages=4,
            **_MINIO,
        )


def test_session_store_accepts_only_known_backends():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, session_store="redis", **_MINIO)


def test_unknown_session_key_is_rejected(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MINIO_ENDPOINT=x\nMINIO_ACCESS_KEY=k\nMINIO_SECRET_KEY=s\nMINIO_BUCKET=b\nSESSION_FOO=1\n"
    )

    with pytest.raises(ValidationError, match="extra"):
        Settings(_env_file=env_file)
