import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.docarag.clients.embedding import EmbeddingGRPCClient
from src.docarag.settings import settings


@pytest.fixture
def mock_embedding_response():
    """Mock embedding response."""
    mock_response = Mock()
    mock_response.embedding = [0.1, 0.2, 0.3] * 128
    return mock_response


@pytest.fixture
def mock_batch_response():
    """Mock batch embedding response."""
    mock_response = Mock()
    embedding_objects = [
        Mock(vector=[0.1, 0.2, 0.3] * 128),
        Mock(vector=[0.4, 0.5, 0.6] * 128),
        Mock(vector=[0.7, 0.8, 0.9] * 128),
    ]
    mock_response.embeddings = embedding_objects
    return mock_response


@pytest.fixture
def mock_dimension_response():
    """Mock dimension response."""
    mock_response = Mock()
    mock_response.dimension = 384
    return mock_response


@pytest.fixture
def mock_async_stub(
    mock_embedding_response, mock_batch_response, mock_dimension_response
):
    """Mock async gRPC stub."""
    stub = Mock()
    stub.EmbedText = AsyncMock(return_value=mock_embedding_response)
    stub.EmbedBatch = AsyncMock(return_value=mock_batch_response)
    stub.GetEmbeddingDimension = AsyncMock(return_value=mock_dimension_response)
    return stub


@pytest.fixture
def mock_sync_stub(
    mock_embedding_response, mock_batch_response, mock_dimension_response
):
    """Mock sync gRPC stub."""
    stub = Mock()
    stub.EmbedText = Mock(return_value=mock_embedding_response)
    stub.EmbedBatch = Mock(return_value=mock_batch_response)
    stub.GetEmbeddingDimension = Mock(return_value=mock_dimension_response)
    return stub


@pytest.fixture
async def async_embedding_client(mock_async_stub):
    """Fixture for async embedding client with mocked stub."""
    with patch("src.docarag.clients.embedding.grpc.aio.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=True)
        client._stub = mock_async_stub
        yield client
        await client.close_async()


@pytest.fixture
def sync_embedding_client(mock_sync_stub):
    """Fixture for sync embedding client with mocked stub."""
    with patch("src.docarag.clients.embedding.grpc.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=False)
        client._stub = mock_sync_stub
        yield client
        client.close()


@pytest.mark.asyncio
async def test_async_single_text_embedding(async_embedding_client):
    """Test async single text embedding generation."""
    text = "This is a test sentence for embedding generation."

    embedding = await async_embedding_client.embed_text_async(text)

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_async_batch_embedding(async_embedding_client):
    """Test async batch embedding generation."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
    ]

    embeddings = await async_embedding_client.embed_batch_async(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) > 0 for emb in embeddings)
    assert all(len(emb) == len(embeddings[0]) for emb in embeddings)


@pytest.mark.asyncio
async def test_async_embedding_dimension(async_embedding_client):
    """Test async embedding dimension query."""
    dimension = await async_embedding_client.get_embedding_dimension_async()

    assert isinstance(dimension, int)
    assert dimension > 0


@pytest.mark.asyncio
async def test_async_empty_text_raises_error():
    """Test that async embedding empty text raises ValueError."""
    with patch("src.docarag.clients.embedding.grpc.aio.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=True)
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            await client.embed_text_async("")
        await client.close_async()


@pytest.mark.asyncio
async def test_async_empty_batch_raises_error():
    """Test that async embedding empty batch raises ValueError."""
    with patch("src.docarag.clients.embedding.grpc.aio.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=True)
        with pytest.raises(ValueError, match="Cannot embed empty list of texts"):
            await client.embed_batch_async([])
        await client.close_async()


@pytest.mark.asyncio
async def test_async_batch_with_only_empty_texts_raises_error():
    """Test that async embedding batch with only empty texts raises ValueError."""
    with patch("src.docarag.clients.embedding.grpc.aio.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=True)
        with pytest.raises(ValueError, match="No valid texts to embed"):
            await client.embed_batch_async(["", "  ", "\n"])
        await client.close_async()


def test_sync_single_text_embedding(sync_embedding_client):
    """Test sync single text embedding generation."""
    text = "This is a test sentence for embedding generation."

    embedding = sync_embedding_client.embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


def test_sync_batch_embedding(sync_embedding_client):
    """Test sync batch embedding generation."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
    ]

    embeddings = sync_embedding_client.embed_batch(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) > 0 for emb in embeddings)
    assert all(len(emb) == len(embeddings[0]) for emb in embeddings)


def test_sync_embedding_dimension(sync_embedding_client):
    """Test sync embedding dimension query."""
    dimension = sync_embedding_client.get_embedding_dimension()

    assert isinstance(dimension, int)
    assert dimension > 0


def test_sync_empty_text_raises_error():
    """Test that sync embedding empty text raises ValueError."""
    with patch("src.docarag.clients.embedding.grpc.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=False)
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            client.embed_text("")
        client.close()


def test_sync_empty_batch_raises_error():
    """Test that sync embedding empty batch raises ValueError."""
    with patch("src.docarag.clients.embedding.grpc.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=False)
        with pytest.raises(ValueError, match="Cannot embed empty list of texts"):
            client.embed_batch([])
        client.close()


def test_sync_batch_with_only_empty_texts_raises_error():
    """Test that sync embedding batch with only empty texts raises ValueError."""
    with patch("src.docarag.clients.embedding.grpc.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=False)
        with pytest.raises(ValueError, match="No valid texts to embed"):
            client.embed_batch(["", "  ", "\n"])
        client.close()


def test_client_initialization_with_defaults():
    """Test client initialization with default settings."""
    client = EmbeddingGRPCClient()

    assert client.url == settings.embedding_service_url
    assert client.timeout == settings.embedding_service_timeout
    assert client.use_async == settings.embedding_use_async


def test_client_initialization_with_custom_values():
    """Test client initialization with custom values."""
    custom_url = "localhost:9999"
    custom_timeout = 60

    client = EmbeddingGRPCClient(
        url=custom_url, timeout=custom_timeout, use_async=False
    )

    assert client.url == custom_url
    assert client.timeout == custom_timeout
    assert client.use_async is False


def test_sync_context_manager():
    """Test sync client as context manager."""
    with patch("src.docarag.clients.embedding.grpc.insecure_channel"):
        with EmbeddingGRPCClient(use_async=False) as client:
            assert client is not None
            assert client._channel is None


@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async client as context manager."""
    with patch("src.docarag.clients.embedding.grpc.aio.insecure_channel"):
        async with EmbeddingGRPCClient(use_async=True) as client:
            assert client is not None
            assert client._channel is None


@pytest.mark.asyncio
async def test_async_embed_text_grpc_error():
    """Test async embed_text handles gRPC errors properly."""
    with patch("src.docarag.clients.embedding.grpc.aio.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=True)
        mock_stub = Mock()
        mock_stub.EmbedText = AsyncMock(side_effect=Exception("gRPC connection error"))
        client._stub = mock_stub

        with pytest.raises(Exception, match="Failed to generate embedding"):
            await client.embed_text_async("test text")

        await client.close_async()


@pytest.mark.asyncio
async def test_async_embed_batch_grpc_error():
    """Test async embed_batch handles gRPC errors properly."""
    with patch("src.docarag.clients.embedding.grpc.aio.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=True)
        mock_stub = Mock()
        mock_stub.EmbedBatch = AsyncMock(side_effect=Exception("gRPC connection error"))
        client._stub = mock_stub

        with pytest.raises(Exception, match="Failed to generate embeddings"):
            await client.embed_batch_async(["test text"])

        await client.close_async()


def test_sync_embed_text_grpc_error():
    """Test sync embed_text handles gRPC errors properly."""
    with patch("src.docarag.clients.embedding.grpc.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=False)
        mock_stub = Mock()
        mock_stub.EmbedText = Mock(side_effect=Exception("gRPC connection error"))
        client._stub = mock_stub

        with pytest.raises(Exception, match="Failed to generate embedding"):
            client.embed_text("test text")

        client.close()


def test_sync_embed_batch_grpc_error():
    """Test sync embed_batch handles gRPC errors properly."""
    with patch("src.docarag.clients.embedding.grpc.insecure_channel"):
        client = EmbeddingGRPCClient(use_async=False)
        mock_stub = Mock()
        mock_stub.EmbedBatch = Mock(side_effect=Exception("gRPC connection error"))
        client._stub = mock_stub

        with pytest.raises(Exception, match="Failed to generate embeddings"):
            client.embed_batch(["test text"])

        client.close()
