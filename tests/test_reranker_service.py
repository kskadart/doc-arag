import grpc
import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.docarag.clients.reranker_client import RerankerGRPCClient
from src.docarag.services.reranker import RerankerService


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
    with patch(
        "src.docarag.services.reranker.RerankerGRPCClient"
    ) as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        service = RerankerService()
        assert service.client == mock_client


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
    """Test that reranking an empty document list short-circuits without a gRPC call."""
    reranked = await reranker_service.rerank_async("query", [], top_k=5)

    assert reranked == []
    mock_grpc_client.rerank_async.assert_not_called()


@pytest.mark.asyncio
async def test_rerank_async_grpc_error_propagates(reranker_service, mock_grpc_client):
    """Test that a gRPC failure from the client propagates through the service."""
    documents = [{"content": "a", "document_name": "a.md"}]
    mock_grpc_client.rerank_async.side_effect = grpc.RpcError("reranker unreachable")

    with pytest.raises(grpc.RpcError):
        await reranker_service.rerank_async("query", documents, top_k=5)


@pytest.mark.asyncio
async def test_close_async(reranker_service, mock_grpc_client):
    """Test that closing the service closes the underlying gRPC client."""
    await reranker_service.close_async()
    mock_grpc_client.close_async.assert_called_once()
