import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from src.docarag.services.vector_db import find_nearest_vectors
from src.docarag.models.responses import VectorSearchResponse


@pytest.fixture
def mock_embedding_client():
    """Create a mock embedding client for testing."""
    client = Mock()
    client.embed_text_async = AsyncMock(return_value=[0.1] * 384)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client for testing."""
    client = Mock()
    client.collections = Mock()
    client.collections.exists = AsyncMock(return_value=True)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_weaviate_response():
    """Create mock Weaviate search response."""
    mock_obj1 = Mock()
    mock_obj1.uuid = "uuid-1"
    mock_obj1.properties = {
        "document_name": "test_doc.pdf",
        "page": 1,
        "content": "This is test content from page 1",
        "date_created": datetime(2024, 1, 1, 12, 0, 0),
    }
    mock_obj1.metadata = Mock()
    mock_obj1.metadata.distance = 0.15

    mock_obj2 = Mock()
    mock_obj2.uuid = "uuid-2"
    mock_obj2.properties = {
        "document_name": "test_doc.pdf",
        "page": 2,
        "content": "This is test content from page 2",
        "date_created": datetime(2024, 1, 1, 12, 0, 0),
    }
    mock_obj2.metadata = Mock()
    mock_obj2.metadata.distance = 0.25

    mock_response = Mock()
    mock_response.objects = [mock_obj1, mock_obj2]
    return mock_response


@pytest.mark.asyncio
async def test_find_nearest_vectors_success(
    mock_embedding_client, mock_weaviate_client, mock_weaviate_response
):
    """Test successful vector search."""
    mock_collection = Mock()
    mock_collection.query = Mock()
    mock_collection.query.near_vector = AsyncMock(return_value=mock_weaviate_response)
    mock_weaviate_client.collections.use = Mock(return_value=mock_collection)

    with patch(
        "src.docarag.services.vector_db.EmbeddingGRPCClient",
        return_value=mock_embedding_client,
    ):
        with patch(
            "src.docarag.services.vector_db.get_vector_db_client",
            return_value=mock_weaviate_client,
        ):
            with patch(
                "src.docarag.services.vector_db.is_collection_exists",
                return_value=True,
            ):
                result = await find_nearest_vectors(
                    query="test query", collection_name="TestCollection", limit=10
                )

    assert isinstance(result, VectorSearchResponse)
    assert result.query == "test query"
    assert result.collection_name == "TestCollection"
    assert result.total_results == 2
    assert len(result.results) == 2

    assert result.results[0].uuid == "uuid-1"
    assert result.results[0].document_name == "test_doc.pdf"
    assert result.results[0].page == 1
    assert result.results[0].content == "This is test content from page 1"
    assert result.results[0].similarity_score == pytest.approx(0.85, rel=0.01)

    assert result.results[1].uuid == "uuid-2"
    assert result.results[1].similarity_score == pytest.approx(0.75, rel=0.01)

    mock_embedding_client.embed_text_async.assert_called_once_with("test query")


@pytest.mark.asyncio
async def test_find_nearest_vectors_collection_not_exists(
    mock_embedding_client, mock_weaviate_client
):
    """Test that non-existent collection raises ValueError."""
    with patch(
        "src.docarag.services.vector_db.is_collection_exists", return_value=False
    ):
        with pytest.raises(
            ValueError, match="Collection 'NonExistentCollection' does not exist"
        ):
            await find_nearest_vectors(
                query="test query", collection_name="NonExistentCollection", limit=10
            )


@pytest.mark.asyncio
async def test_find_nearest_vectors_with_limit(
    mock_embedding_client, mock_weaviate_client, mock_weaviate_response
):
    """Test that limit parameter is passed correctly."""
    mock_collection = Mock()
    mock_collection.query = Mock()
    mock_collection.query.near_vector = AsyncMock(return_value=mock_weaviate_response)
    mock_weaviate_client.collections.use = Mock(return_value=mock_collection)

    with patch(
        "src.docarag.services.vector_db.EmbeddingGRPCClient",
        return_value=mock_embedding_client,
    ):
        with patch(
            "src.docarag.services.vector_db.get_vector_db_client",
            return_value=mock_weaviate_client,
        ):
            with patch(
                "src.docarag.services.vector_db.is_collection_exists",
                return_value=True,
            ):
                await find_nearest_vectors(
                    query="test query", collection_name="TestCollection", limit=20
                )


@pytest.mark.asyncio
async def test_find_nearest_vectors_empty_results(
    mock_embedding_client, mock_weaviate_client
):
    """Test vector search with no results."""
    mock_collection = Mock()
    mock_collection.query = Mock()
    mock_empty_response = Mock()
    mock_empty_response.objects = []
    mock_collection.query.near_vector = AsyncMock(return_value=mock_empty_response)
    mock_weaviate_client.collections.use = Mock(return_value=mock_collection)

    with patch(
        "src.docarag.services.vector_db.EmbeddingGRPCClient",
        return_value=mock_embedding_client,
    ):
        with patch(
            "src.docarag.services.vector_db.get_vector_db_client",
            return_value=mock_weaviate_client,
        ):
            with patch(
                "src.docarag.services.vector_db.is_collection_exists",
                return_value=True,
            ):
                result = await find_nearest_vectors(
                    query="test query", collection_name="TestCollection", limit=10
                )

    assert isinstance(result, VectorSearchResponse)
    assert result.total_results == 0
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_find_nearest_vectors_embedding_failure(mock_weaviate_client):
    """Test handling of embedding service failure."""
    mock_embedding_client = Mock()
    mock_embedding_client.embed_text_async = AsyncMock(
        side_effect=Exception("Embedding service unavailable")
    )
    mock_embedding_client.__aenter__ = AsyncMock(return_value=mock_embedding_client)
    mock_embedding_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "src.docarag.services.vector_db.EmbeddingGRPCClient",
        return_value=mock_embedding_client,
    ):
        with patch(
            "src.docarag.services.vector_db.is_collection_exists", return_value=True
        ):
            with pytest.raises(Exception, match="Embedding service unavailable"):
                await find_nearest_vectors(
                    query="test query", collection_name="TestCollection", limit=10
                )
