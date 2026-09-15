"""Tests for the Cohere/Jina-style HTTP reranker client (mocked transport)."""

import json

import httpx
import pytest

from src.docarag.clients.reranker_http import RerankerHTTPClient, RerankResult
from src.docarag.errors import RerankerError


def _make_client(handler, **kwargs) -> RerankerHTTPClient:
    return RerankerHTTPClient(
        base_url="http://rerank.test/v1",
        api_key="secret",
        model="test/reranker",
        retry_backoff=0,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_rerank_async_posts_expected_payload_and_parses_results():
    """Test the request shape and the parsed (index, score) results."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.3},
                ]
            },
        )

    async with _make_client(handler) as client:
        results = await client.rerank_async("q", ["a", "b", "c"], top_n=2)

    assert results == [RerankResult(2, 0.9), RerankResult(0, 0.3)]
    assert seen["url"] == "http://rerank.test/v1/rerank"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"] == {
        "model": "test/reranker",
        "query": "q",
        "documents": ["a", "b", "c"],
        "top_n": 2,
    }


@pytest.mark.asyncio
async def test_empty_query_or_texts_raise_value_error():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    async with _make_client(handler) as client:
        with pytest.raises(ValueError):
            await client.rerank_async("", ["a"], top_n=1)
        with pytest.raises(ValueError):
            await client.rerank_async("q", [], top_n=1)


@pytest.mark.asyncio
async def test_non_success_status_raises_reranker_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="no such model")

    async with _make_client(handler) as client:
        with pytest.raises(RerankerError, match="404"):
            await client.rerank_async("q", ["a"], top_n=1)


@pytest.mark.asyncio
async def test_server_error_exhausts_retries():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502, text="bad gateway")

    async with _make_client(handler, max_retries=1) as client:
        with pytest.raises(RerankerError, match="after 2 attempts"):
            await client.rerank_async("q", ["a"], top_n=1)

    assert calls == 2


@pytest.mark.asyncio
async def test_timeout_raises_reranker_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    async with _make_client(handler, max_retries=0) as client:
        with pytest.raises(RerankerError, match="slow"):
            await client.rerank_async("q", ["a"], top_n=1)


@pytest.mark.asyncio
async def test_missing_results_or_bad_index_raise_reranker_error():
    def no_results(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    async with _make_client(no_results) as client:
        with pytest.raises(RerankerError, match="Malformed"):
            await client.rerank_async("q", ["a"], top_n=1)

    def bad_index(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"results": [{"index": 5, "relevance_score": 1}]}
        )

    async with _make_client(bad_index) as client:
        with pytest.raises(RerankerError, match="out of range"):
            await client.rerank_async("q", ["a"], top_n=1)


@pytest.mark.asyncio
async def test_sglang_bare_list_response_is_parsed():
    """Test the SGLang dialect: top-level list with `score` instead of `relevance_score`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"index": 1, "score": 0.8, "document": "b", "meta_info": {}},
                {"index": 0, "score": 0.1, "document": "a", "meta_info": {}},
            ],
        )

    async with _make_client(handler) as client:
        results = await client.rerank_async("q", ["a", "b"], top_n=2)

    assert results == [RerankResult(1, 0.8), RerankResult(0, 0.1)]
