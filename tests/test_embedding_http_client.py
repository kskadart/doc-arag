"""Tests for the OpenAI-compatible HTTP embedding client (mocked transport)."""

import json

import httpx
import pytest

from src.docarag.clients.embedding_http import EmbeddingHTTPClient
from src.docarag.errors import EmbeddingError


def _embeddings_payload(count: int, dim: int = 4, shuffle: bool = False) -> dict:
    data = [
        {"object": "embedding", "index": i, "embedding": [float(i)] * dim}
        for i in range(count)
    ]
    if shuffle:
        data.reverse()
    return {"object": "list", "data": data, "model": "test/embedding-model"}


def _make_client(handler, **kwargs) -> EmbeddingHTTPClient:
    return EmbeddingHTTPClient(
        base_url="http://embed.test/v1",
        api_key="secret",
        model="test/embedding-model",
        retry_backoff=0,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_embed_text_async_posts_openai_payload():
    """Test the request shape: URL, bearer header, model and input."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_embeddings_payload(1))

    async with _make_client(handler) as client:
        vector = await client.embed_text_async("hello")

    assert vector == [0.0, 0.0, 0.0, 0.0]
    assert seen["url"] == "http://embed.test/v1/embeddings"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"] == {
        "model": "test/embedding-model",
        "input": ["hello"],
        "encoding_format": "float",
    }


@pytest.mark.asyncio
async def test_embed_batch_async_splits_into_batches_preserving_order():
    """Test that 25 texts with batch_size 10 make 3 requests and keep order."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        calls.append(len(inputs))
        return httpx.Response(200, json=_embeddings_payload(len(inputs)))

    async with _make_client(handler, batch_size=10) as client:
        vectors = await client.embed_batch_async([f"t{i}" for i in range(25)])

    assert calls == [10, 10, 5]
    assert len(vectors) == 25
    assert vectors[10][0] == 0.0  # first item of the second batch


@pytest.mark.asyncio
async def test_response_items_are_sorted_by_index():
    """Test that out-of-order `data` items are put back into input order."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_embeddings_payload(3, shuffle=True))

    async with _make_client(handler) as client:
        vectors = await client.embed_batch_async(["a", "b", "c"])

    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


@pytest.mark.asyncio
async def test_dimensions_sent_only_when_configured():
    """Test that the optional `dimensions` field is present only when set."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_embeddings_payload(1))

    async with _make_client(handler) as client:
        await client.embed_text_async("x")
    async with _make_client(handler, dimensions=256) as client:
        await client.embed_text_async("x")

    assert "dimensions" not in bodies[0]
    assert bodies[1]["dimensions"] == 256


@pytest.mark.asyncio
async def test_empty_inputs_raise_value_error():
    """Test that empty text or empty list are rejected before any request."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    async with _make_client(handler) as client:
        with pytest.raises(ValueError):
            await client.embed_text_async("   ")
        with pytest.raises(ValueError):
            await client.embed_batch_async([])


@pytest.mark.asyncio
async def test_client_error_status_raises_embedding_error_without_retry():
    """Test that a 4xx answer fails immediately."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "bad key"})

    async with _make_client(handler, max_retries=3) as client:
        with pytest.raises(EmbeddingError, match="401"):
            await client.embed_text_async("x")

    assert calls == 1


@pytest.mark.asyncio
async def test_server_error_is_retried_then_succeeds():
    """Test that a 503 followed by 200 succeeds after one retry."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=_embeddings_payload(1))

    async with _make_client(handler, max_retries=1) as client:
        vector = await client.embed_text_async("x")

    assert calls == 2
    assert len(vector) == 4


@pytest.mark.asyncio
async def test_server_error_exhausts_retries():
    """Test that persistent 5xx ends in EmbeddingError after max_retries + 1 attempts."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="boom")

    async with _make_client(handler, max_retries=2) as client:
        with pytest.raises(EmbeddingError, match="after 3 attempts"):
            await client.embed_text_async("x")

    assert calls == 3


@pytest.mark.asyncio
async def test_transport_error_raises_embedding_error():
    """Test that connection failures are wrapped into EmbeddingError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _make_client(handler, max_retries=0) as client:
        with pytest.raises(EmbeddingError, match="connection refused"):
            await client.embed_text_async("x")


@pytest.mark.asyncio
async def test_malformed_payload_raises_embedding_error():
    """Test that a payload without `data` or with wrong counts is rejected."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    async with _make_client(handler) as client:
        with pytest.raises(EmbeddingError, match="Malformed"):
            await client.embed_text_async("x")

    def short_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_embeddings_payload(1))

    async with _make_client(short_handler) as client:
        with pytest.raises(EmbeddingError, match="1 vectors for 2 inputs"):
            await client.embed_batch_async(["a", "b"])


@pytest.mark.asyncio
async def test_get_embedding_dimension_probes_once():
    """Test that the dimension probe is cached."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_embeddings_payload(1, dim=7))

    async with _make_client(handler) as client:
        assert await client.get_embedding_dimension_async() == 7
        assert await client.get_embedding_dimension_async() == 7

    assert calls == 1


@pytest.mark.asyncio
async def test_extra_body_is_merged_but_cannot_override_core_fields():
    """Provider routing reaches the request body; model and input stay authoritative."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [0.5, 0.25]}]}
        )

    client = EmbeddingHTTPClient(
        base_url="http://embed.test/v1",
        api_key="k",
        model="qwen/qwen3-embedding-8b",
        max_retries=0,
        transport=httpx.MockTransport(handler),
        extra_body={
            "provider": {"order": ["nebius"], "allow_fallbacks": False},
            "model": "ignored",
        },
    )

    assert await client.embed_text_async("hello") == [0.5, 0.25]
    assert seen[0]["provider"] == {"order": ["nebius"], "allow_fallbacks": False}
    assert seen[0]["model"] == "qwen/qwen3-embedding-8b"
    assert seen[0]["input"] == ["hello"]
