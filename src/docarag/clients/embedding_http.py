"""HTTP client for any OpenAI-compatible `/embeddings` endpoint."""

import asyncio
import logging
from typing import Any

import httpx

from src.docarag.errors import EmbeddingError
from src.docarag.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingHTTPClient:
    """Client for `POST {base_url}/embeddings` (OpenRouter, Ollama, vLLM, TEI, DashScope)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        batch_size: int | None = None,
        dimensions: int | None = None,
        max_retries: int | None = None,
        retry_backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the client; every argument falls back to settings.

        Args:
            base_url: Endpoint base ending in `/v1` (the `/embeddings` path is appended)
            api_key: Bearer token, "EMPTY" for servers without auth
            model: Embedding model identifier
            timeout: Request timeout in seconds
            batch_size: Maximum texts per request
            dimensions: Optional output dimension, sent only when set
            max_retries: Retries on transport errors, 429 and 5xx responses
            retry_backoff: Base of the exponential backoff in seconds
            transport: Optional httpx transport, used by tests to mock the server
            extra_body: Extra JSON merged into every request body (provider routing);
                the model, input and dimensions fields always win
        """
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.api_key = api_key or settings.embedding_api_key.get_secret_value()
        self.model = model or settings.embedding_model
        self.timeout = timeout or settings.embedding_timeout
        self.batch_size = batch_size or settings.embedding_batch_size
        self.dimensions = (
            dimensions if dimensions is not None else settings.embedding_dimensions
        )
        self.max_retries = (
            max_retries if max_retries is not None else settings.embedding_max_retries
        )
        self.retry_backoff = retry_backoff
        self.extra_body = (
            extra_body if extra_body is not None else settings.embedding_extra_body
        )
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._dimension: int | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the pooled HTTP client."""
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

    async def embed_text_async(self, text: str) -> list[float]:
        """
        Embed a single text.

        Raises:
            ValueError: If the text is empty
            EmbeddingError: If the endpoint fails or answers with an unusable payload
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        vectors = await self._embed_request([text])
        return vectors[0]

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """
        Embed many texts, `batch_size` per request, preserving order.

        Raises:
            ValueError: If the list is empty
            EmbeddingError: If any request fails
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                await self._embed_request(texts[start : start + self.batch_size])
            )
        return vectors

    async def get_embedding_dimension_async(self) -> int:
        """Probe the endpoint once and cache the vector dimension."""
        if self._dimension is None:
            probe = await self.embed_text_async("dimension probe")
            self._dimension = len(probe)
        return self._dimension

    async def _embed_request(self, inputs: list[str]) -> list[list[float]]:
        """Issue one `/embeddings` call with retries and validate the payload."""
        payload: dict[str, Any] = {
            **(self.extra_body or {}),
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                await asyncio.sleep(min(self.retry_backoff * 2 ** (attempt - 1), 8.0))
            try:
                response = await client.post("/embeddings", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    f"Embedding request failed (attempt {attempt + 1}): {exc}"
                )
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_error = EmbeddingError(
                    f"Embedding endpoint returned {response.status_code}: {response.text[:200]}"
                )
                logger.warning(str(last_error))
                continue
            if response.status_code >= 400:
                raise EmbeddingError(
                    f"Embedding endpoint returned {response.status_code}: {response.text[:200]}"
                )
            return self._parse_response(response, expected=len(inputs))

        raise EmbeddingError(
            f"Embedding request failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _parse_response(
        self, response: httpx.Response, expected: int
    ) -> list[list[float]]:
        """Extract vectors in input order from an OpenAI-style embeddings payload."""
        try:
            items = response.json()["data"]
            ordered = sorted(items, key=lambda item: int(item["index"]))
            vectors = [[float(x) for x in item["embedding"]] for item in ordered]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(f"Malformed embeddings payload: {exc}") from exc

        if len(vectors) != expected:
            raise EmbeddingError(
                f"Embedding endpoint returned {len(vectors)} vectors for {expected} inputs"
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise EmbeddingError(
                f"Inconsistent embedding dimensions in one batch: {dimensions}"
            )
        return vectors

    async def close_async(self) -> None:
        """Close the pooled HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "EmbeddingHTTPClient":
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        await self.close_async()
