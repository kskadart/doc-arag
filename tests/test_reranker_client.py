import grpc
import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.docarag.clients.reranker_client import RerankerGRPCClient
from src.docarag.settings import settings


@pytest.fixture
def mock_rerank_response():
    """Mock rerank response, scores aligned with the request texts order."""
    mock_response = Mock()
    mock_response.scores = [0.2, 0.9, 0.5]
    return mock_response


@pytest.fixture
def mock_health_response():
    """Mock health check response."""
    mock_response = Mock()
    mock_response.status = "SERVING"
    mock_response.model_name = "BAAI/bge-reranker-v2-m3"
    mock_response.device = "cpu"
    return mock_response


@pytest.fixture
def mock_stub(mock_rerank_response, mock_health_response):
    """Mock async gRPC stub."""
    stub = Mock()
    stub.Rerank = AsyncMock(return_value=mock_rerank_response)
    stub.HealthCheck = AsyncMock(return_value=mock_health_response)
    return stub


@pytest.fixture
async def reranker_client(mock_stub):
    """Fixture for reranker client with a mocked stub."""
    with patch("src.docarag.clients.reranker_client.grpc.aio.insecure_channel"):
        client = RerankerGRPCClient()
        client._stub = mock_stub
        yield client
        await client.close_async()


@pytest.mark.asyncio
async def test_rerank_async_returns_scores_in_texts_order(reranker_client, mock_stub):
    """Test that rerank_async returns scores aligned with the input texts order."""
    scores = await reranker_client.rerank_async(
        "query", ["first", "second", "third"]
    )

    assert scores == [0.2, 0.9, 0.5]
    call_args = mock_stub.Rerank.call_args
    request = call_args.args[0]
    assert request.query == "query"
    assert list(request.texts) == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_rerank_async_empty_query_raises_value_error():
    """Test that reranking with an empty query raises ValueError."""
    with patch("src.docarag.clients.reranker_client.grpc.aio.insecure_channel"):
        client = RerankerGRPCClient()
        with pytest.raises(ValueError, match="Cannot rerank with an empty query"):
            await client.rerank_async("", ["text"])
        await client.close_async()


@pytest.mark.asyncio
async def test_rerank_async_empty_texts_raises_value_error():
    """Test that reranking an empty list of texts raises ValueError."""
    with patch("src.docarag.clients.reranker_client.grpc.aio.insecure_channel"):
        client = RerankerGRPCClient()
        with pytest.raises(ValueError, match="Cannot rerank an empty list of texts"):
            await client.rerank_async("query", [])
        await client.close_async()


@pytest.mark.asyncio
async def test_rerank_async_grpc_error_propagates(reranker_client, mock_stub):
    """Test that a gRPC failure from the reranker service propagates to the caller."""
    mock_stub.Rerank = AsyncMock(side_effect=grpc.RpcError("reranker unreachable"))

    with pytest.raises(grpc.RpcError):
        await reranker_client.rerank_async("query", ["text"])


@pytest.mark.asyncio
async def test_health_check_async_returns_service_status(reranker_client):
    """Test that health_check_async returns the reranker service health response."""
    response = await reranker_client.health_check_async()

    assert response.status == "SERVING"
    assert response.model_name == "BAAI/bge-reranker-v2-m3"
    assert response.device == "cpu"


def test_client_initialization_with_defaults():
    """Test that the client reads url and timeout from settings by default."""
    client = RerankerGRPCClient()

    assert client.url == settings.reranker_service_url
    assert client.timeout == settings.reranker_timeout


def test_client_initialization_with_custom_values():
    """Test that explicit url and timeout override the settings defaults."""
    client = RerankerGRPCClient(url="localhost:9999", timeout=5)

    assert client.url == "localhost:9999"
    assert client.timeout == 5


@pytest.mark.asyncio
async def test_async_context_manager_closes_channel_on_exit():
    """Test that the client behaves as an async context manager and resets its channel."""
    with patch("src.docarag.clients.reranker_client.grpc.aio.insecure_channel"):
        async with RerankerGRPCClient() as client:
            assert client is not None
            assert client._channel is None
