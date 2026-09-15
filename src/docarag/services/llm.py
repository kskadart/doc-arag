"""Chat model factory: one place that knows which LLM provider is configured."""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.docarag.settings import settings


@lru_cache(maxsize=8)
def get_chat_model(temperature: float) -> BaseChatModel:
    """
    Build (and cache per temperature) the chat model for the configured provider.

    Args:
        temperature: Sampling temperature for this model instance

    Returns:
        A LangChain chat model; every node calls `ainvoke(str)` and reads `.text`

    Raises:
        ValueError: If `settings.llm_provider` is unknown
    """
    if settings.llm_provider == "openai":
        return ChatOpenAI(  # type: ignore[call-arg]
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=temperature,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            extra_body=settings.llm_extra_body,
            default_headers=settings.llm_default_headers,
        )

    if settings.llm_provider == "anthropic":
        # langchain-anthropic declares its optional "timeout" and "stop" aliases as
        # Field(None, ...), which type checkers do not read as a default
        return ChatAnthropic(  # type: ignore[call-arg]
            model_name=settings.anthropic_model or "",
            api_key=SecretStr(settings.anthropic_api_key or ""),
            temperature=temperature,
            anthropic_proxy=settings.anthropic_proxy,
        )

    raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")
