import pytest
from unittest.mock import Mock, patch
from src.docarag.services.embeddings import EmbeddingService
from src.docarag.clients.embedding import EmbeddingGRPCClient


@pytest.fixture
def mock_grpc_client():
    """Create a mock gRPC client for testing."""
    client = Mock(spec=EmbeddingGRPCClient)
    client.embed_text.return_value = [0.1] * 384
    client.embed_batch.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    client.get_embedding_dimension.return_value = 384
    client.embed_text_async.return_value = [0.1] * 384
    client.embed_batch_async.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    client.get_embedding_dimension_async.return_value = 384
    return client


@pytest.fixture
def embedding_service(mock_grpc_client):
    """Create an embedding service instance for testing."""
    return EmbeddingService(client=mock_grpc_client)


def test_embedding_service_initialization():
    """Test that embedding service initializes with gRPC client."""
    with patch(
        "src.docarag.services.embeddings.EmbeddingGRPCClient"
    ) as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        service = EmbeddingService()
        assert service.client == mock_client


def test_embed_text(embedding_service, mock_grpc_client):
    """Test embedding a single text."""
    text = "This is a test sentence."
    embedding = embedding_service.embed_text(text)

    mock_grpc_client.embed_text.assert_called_once_with(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)


def test_embed_text_empty(embedding_service, mock_grpc_client):
    """Test embedding empty text raises error."""
    mock_grpc_client.embed_text.side_effect = ValueError("Cannot embed empty text")

    with pytest.raises(ValueError, match="Cannot embed empty text"):
        embedding_service.embed_text("")


def test_embed_batch(embedding_service, mock_grpc_client):
    """Test embedding multiple texts."""
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = embedding_service.embed_batch(texts)

    mock_grpc_client.embed_batch.assert_called_once_with(
        texts, batch_size=32, max_length=None, normalize=None, pooling_strategy=None
    )
    assert len(embeddings) == len(texts)
    assert all(len(emb) == 384 for emb in embeddings)


def test_embed_batch_empty(embedding_service, mock_grpc_client):
    """Test embedding empty list raises error."""
    mock_grpc_client.embed_batch.side_effect = ValueError(
        "Cannot embed empty list of texts"
    )

    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedding_service.embed_batch([])


def test_get_embedding_dimension(embedding_service, mock_grpc_client):
    """Test getting embedding dimension."""
    dim = embedding_service.get_embedding_dimension()

    mock_grpc_client.get_embedding_dimension.assert_called_once()
    assert isinstance(dim, int)
    assert dim == 384


@pytest.mark.asyncio
async def test_embed_text_async(embedding_service, mock_grpc_client):
    """Test async embedding a single text."""
    text = "This is a test sentence."
    embedding = await embedding_service.embed_text_async(text)

    mock_grpc_client.embed_text_async.assert_called_once_with(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 384


@pytest.mark.asyncio
async def test_embed_batch_async(embedding_service, mock_grpc_client):
    """Test async embedding multiple texts."""
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = await embedding_service.embed_batch_async(texts)

    mock_grpc_client.embed_batch_async.assert_called_once_with(
        texts, batch_size=32, max_length=None, normalize=None, pooling_strategy=None
    )
    assert len(embeddings) == len(texts)
    assert all(len(emb) == 384 for emb in embeddings)


@pytest.mark.asyncio
async def test_get_embedding_dimension_async(embedding_service, mock_grpc_client):
    """Test async getting embedding dimension."""
    dim = await embedding_service.get_embedding_dimension_async()

    mock_grpc_client.get_embedding_dimension_async.assert_called_once()
    assert isinstance(dim, int)
    assert dim == 384


def test_close_sync(embedding_service, mock_grpc_client):
    """Test closing sync client."""
    embedding_service.close()
    mock_grpc_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_async(embedding_service, mock_grpc_client):
    """Test closing async client."""
    await embedding_service.close_async()
    mock_grpc_client.close_async.assert_called_once()
