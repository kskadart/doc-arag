"""HTTP client for `POST {base_url}/rerank` endpoints.

Accepts both response dialects seen in the wild:
- OpenRouter / llama.cpp / Cohere / Jina / vLLM: ``{"results": [{"index", "relevance_score"}]}``
- SGLang: a bare list ``[{"index", "score", "document"}]``
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from src.docarag.errors import RerankerError
from src.docarag.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    """One scored candidate: position in the input list and its relevance."""

    index: int
    relevance_score: float


class RerankerHTTPClient:
    """Client for OpenAI-style rerank APIs returning `results[{index, relevance_score}]`."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        base = base_url or settings.reranker_base_url
        if not base:
            raise ValueError("reranker_base_url is required for the HTTP reranker")
        self.base_url = base.rstrip("/")
        self.api_key = api_key or settings.reranker_api_key.get_secret_value()
        self.model = model or settings.reranker_model
        self.timeout = timeout or settings.reranker_timeout
        self.max_retries = (
            max_retries if max_retries is not None else settings.reranker_max_retries
        )
        self.retry_backoff = retry_backoff
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
                transport=self._transport,
            )
        return self._client

    async def rerank_async(
        self, query: str, texts: list[str], top_n: int
    ) -> list[RerankResult]:
        """
        Score candidates against the query.

        Returns:
            Up to `top_n` results; the server may already truncate and sort them

        Raises:
            ValueError: If query or texts are empty
            RerankerError: On transport failure, non-2xx status or malformed payload
        """
        if not query or not query.strip():
            raise ValueError("Cannot rerank with an empty query")
        if not texts:
            raise ValueError("Cannot rerank an empty list of texts")

        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": top_n,
        }
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                await asyncio.sleep(min(self.retry_backoff * 2 ** (attempt - 1), 8.0))
            try:
                response = await client.post("/rerank", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(f"Rerank request failed (attempt {attempt + 1}): {exc}")
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_error = RerankerError(
                    f"Rerank endpoint returned {response.status_code}: {response.text[:200]}"
                )
                logger.warning(str(last_error))
                continue
            if response.status_code >= 400:
                raise RerankerError(
                    f"Rerank endpoint returned {response.status_code}: {response.text[:200]}"
                )
            return self._parse_response(response, candidates=len(texts))

        raise RerankerError(
            f"Rerank request failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _parse_response(
        response: httpx.Response, candidates: int
    ) -> list[RerankResult]:
        try:
            payload = response.json()
            items = payload["results"] if isinstance(payload, dict) else payload
            results = []
            for item in items:
                score = (
                    item["relevance_score"]
                    if "relevance_score" in item
                    else item["score"]
                )
                results.append(RerankResult(int(item["index"]), float(score)))
        except (ValueError, KeyError, TypeError) as exc:
            raise RerankerError(f"Malformed rerank payload: {exc}") from exc

        for result in results:
            if not 0 <= result.index < candidates:
                raise RerankerError(
                    f"Rerank result index {result.index} out of range for {candidates} texts"
                )
        return results

    async def close_async(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "RerankerHTTPClient":
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        await self.close_async()
