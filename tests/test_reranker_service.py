import grpc
import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.docarag.clients.reranker_client import RerankerGRPCClient
from src.docarag.clients.reranker_http import RerankerHTTPClient, RerankResult
from src.docarag.errors import RerankerError
from src.docarag.services.reranker import RerankerService, build_reranker_client
from src.docarag.settings import settings


@pytest.fixture
def mock_grpc_client():
    """Create a mock gRPC client for testing."""
    client = Mock(spec=RerankerGRPCClient)
    client.rerank_async = AsyncMock()
    client.close_async = AsyncMock()
    return client


@pytest.fixture
def reranker_service(mock_grpc_client):
    """Create a reranker service instance with an injected mock client."""
    return RerankerService(client=mock_grpc_client)


def test_reranker_service_initialization():
    """Test that the reranker service creates a gRPC client when none is provided."""
    with patch("src.docarag.services.reranker.RerankerGRPCClient") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        service = RerankerService()
        assert service.client == mock_client


def test_build_reranker_client_follows_provider_setting(monkeypatch):
    """Test that the provider setting selects gRPC, HTTP or nothing."""
    monkeypatch.setattr(settings, "reranker_provider", "none")
    assert build_reranker_client() is None

    monkeypatch.setattr(settings, "reranker_provider", "openai-rerank")
    monkeypatch.setattr(settings, "reranker_base_url", "http://rerank.test/v1")
    assert isinstance(build_reranker_client(), RerankerHTTPClient)

    monkeypatch.setattr(settings, "reranker_provider", "grpc")
    assert isinstance(build_reranker_client(), RerankerGRPCClient)


@pytest.mark.asyncio
async def test_rerank_async_sorts_documents_by_score_descending(
    reranker_service, mock_grpc_client
):
    """Test that documents are reordered by descending relevance score."""
    documents = [
        {"content": "low relevance", "document_name": "a.md"},
        {"content": "high relevance", "document_name": "b.md"},
        {"content": "mid relevance", "document_name": "c.md"},
    ]
    mock_grpc_client.rerank_async.return_value = [0.1, 0.9, 0.5]

    reranked = await reranker_service.rerank_async("query", documents, top_k=3)

    assert [doc["document_name"] for doc in reranked] == ["b.md", "c.md", "a.md"]
    assert [doc["rerank_score"] for doc in reranked] == [0.9, 0.5, 0.1]


@pytest.mark.asyncio
async def test_rerank_async_applies_top_k_limit(reranker_service, mock_grpc_client):
    """Test that only the top_k highest scoring documents are returned."""
    documents = [
        {"content": "a", "document_name": "a.md"},
        {"content": "b", "document_name": "b.md"},
        {"content": "c", "document_name": "c.md"},
    ]
    mock_grpc_client.rerank_async.return_value = [0.3, 0.9, 0.6]

    reranked = await reranker_service.rerank_async("query", documents, top_k=2)

    assert len(reranked) == 2
    assert [doc["document_name"] for doc in reranked] == ["b.md", "c.md"]


@pytest.mark.asyncio
async def test_rerank_async_empty_documents_returns_empty_list(
    reranker_service, mock_grpc_client
):
    """Test that reranking an empty document list short-circuits without a call."""
    reranked = await reranker_service.rerank_async("query", [], top_k=5)

    assert reranked == []
    mock_grpc_client.rerank_async.assert_not_called()


@pytest.mark.asyncio
async def test_rerank_async_grpc_error_becomes_reranker_error(
    reranker_service, mock_grpc_client
):
    """Test that a gRPC failure is normalised into RerankerError."""
    documents = [{"content": "a", "document_name": "a.md"}]
    mock_grpc_client.rerank_async.side_effect = grpc.RpcError("reranker unreachable")

    with pytest.raises(RerankerError, match="reranker unreachable"):
        await reranker_service.rerank_async("query", documents, top_k=5)


@pytest.mark.asyncio
async def test_rerank_async_grpc_score_count_mismatch_raises(
    reranker_service, mock_grpc_client
):
    """Test that a wrong-length score list is rejected instead of mis-zipped."""
    documents = [{"content": "a"}, {"content": "b"}]
    mock_grpc_client.rerank_async.return_value = [0.5]

    with pytest.raises(RerankerError, match="1 scores for 2 texts"):
        await reranker_service.rerank_async("query", documents, top_k=5)


@pytest.mark.asyncio
async def test_rerank_async_maps_http_result_index_back_to_document():
    """Test that HTTP (index, score) results are mapped onto the right documents."""
    client = Mock(spec=RerankerHTTPClient)
    client.rerank_async = AsyncMock(
        return_value=[RerankResult(2, 0.95), RerankResult(0, 0.2)]
    )
    service = RerankerService(client=client)
    documents = [{"content": "a"}, {"content": "b"}, {"content": "c"}]

    reranked = await service.rerank_async("query", documents, top_k=5)

    assert [doc["content"] for doc in reranked] == ["c", "a"]
    assert reranked[0]["rerank_score"] == 0.95
    client.rerank_async.assert_awaited_once_with("query", ["a", "b", "c"], 5)


@pytest.mark.asyncio
async def test_rerank_async_out_of_range_index_raises():
    """Test that an index outside the document list is an error."""
    client = Mock(spec=RerankerHTTPClient)
    client.rerank_async = AsyncMock(return_value=[RerankResult(7, 0.5)])
    service = RerankerService(client=client)

    with pytest.raises(RerankerError, match="out of range"):
        await service.rerank_async("query", [{"content": "a"}], top_k=5)


@pytest.mark.asyncio
async def test_rerank_async_disabled_provider_raises():
    """Test that a service without a client reports the reranker as disabled."""
    with patch(
        "src.docarag.services.reranker.build_reranker_client", return_value=None
    ):
        service = RerankerService()

    with pytest.raises(RerankerError, match="disabled"):
        await service.rerank_async("query", [{"content": "a"}], top_k=5)


@pytest.mark.asyncio
async def test_close_async(reranker_service, mock_grpc_client):
    """Test that closing the service closes the underlying client."""
    await reranker_service.close_async()
    mock_grpc_client.close_async.assert_called_once()
