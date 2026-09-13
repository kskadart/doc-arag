import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.docarag.clients.embedding_http import EmbeddingHTTPClient
from src.docarag.services.embeddings import EmbeddingService


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP embedding client for testing."""
    client = Mock(spec=EmbeddingHTTPClient)
    client.embed_text_async = AsyncMock(return_value=[0.1] * 384)
    client.embed_batch_async = AsyncMock(
        return_value=[[0.1] * 384, [0.2] * 384, [0.3] * 384]
    )
    client.get_embedding_dimension_async = AsyncMock(return_value=384)
    client.close_async = AsyncMock()
    return client


@pytest.fixture
def embedding_service(mock_http_client):
    """Create an embedding service instance for testing."""
    return EmbeddingService(client=mock_http_client)


def test_embedding_service_initialization():
    """Test that embedding service builds an HTTP client from settings."""
    with patch(
        "src.docarag.services.embeddings.EmbeddingHTTPClient"
    ) as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        service = EmbeddingService()
        assert service.client == mock_client


@pytest.mark.asyncio
async def test_embed_text_async(embedding_service, mock_http_client):
    """Test async embedding a single text."""
    embedding = await embedding_service.embed_text_async("This is a test sentence.")

    mock_http_client.embed_text_async.assert_awaited_once_with(
        "This is a test sentence."
    )
    assert len(embedding) == 384


@pytest.mark.asyncio
async def test_embed_batch_async(embedding_service, mock_http_client):
    """Test async embedding multiple texts."""
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = await embedding_service.embed_batch_async(texts)

    mock_http_client.embed_batch_async.assert_awaited_once_with(texts)
    assert len(embeddings) == len(texts)


@pytest.mark.asyncio
async def test_get_embedding_dimension_async(embedding_service, mock_http_client):
    """Test async getting embedding dimension."""
    assert await embedding_service.get_embedding_dimension_async() == 384
    mock_http_client.get_embedding_dimension_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_async(embedding_service, mock_http_client):
    """Test closing the client."""
    await embedding_service.close_async()
    mock_http_client.close_async.assert_awaited_once()
