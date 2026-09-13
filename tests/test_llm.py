"""Tests for the chat model factory."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.docarag.services.llm import get_chat_model
from src.docarag.settings import settings


@pytest.fixture(autouse=True)
def clear_cache():
    get_chat_model.cache_clear()
    yield
    get_chat_model.cache_clear()


def test_openai_provider_builds_chat_openai(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_model", "qwen/test")
    monkeypatch.setattr(settings, "llm_base_url", "http://llm.test/v1")
    monkeypatch.setattr(settings, "llm_extra_body", {"reasoning": {"enabled": False}})
    monkeypatch.setattr(settings, "llm_default_headers", {"X-Title": "doc-arag"})

    model = get_chat_model(0.3)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "qwen/test"
    assert model.openai_api_base == "http://llm.test/v1"
    assert model.temperature == 0.3
    assert model.extra_body == {"reasoning": {"enabled": False}}
    assert model.default_headers == {"X-Title": "doc-arag"}


def test_anthropic_provider_builds_chat_anthropic(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_model", "claude-sonnet-5")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")

    model = get_chat_model(0.5)

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-5"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mystery")

    with pytest.raises(ValueError, match="Unknown llm_provider"):
        get_chat_model(0.1)


def test_models_are_cached_per_temperature(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")

    assert get_chat_model(0.3) is get_chat_model(0.3)
    assert get_chat_model(0.3) is not get_chat_model(0.7)
