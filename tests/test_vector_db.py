import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from weaviate.collections.classes.batch import DeleteManyReturn
from weaviate.exceptions import (
    UnexpectedStatusCodeError,
    WeaviateInsertManyAllFailedError,
)
from src.docarag.errors import EmbeddingError
from src.docarag.services.vector_db import (
    add_batch_objects,
    create_default_collection,
    delete_objects_by_document_name,
    find_nearest_vectors,
    verify_embedding_dimension,
)
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
        "src.docarag.services.vector_db.get_embedding_service",
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
        "src.docarag.services.vector_db.get_embedding_service",
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
        "src.docarag.services.vector_db.get_embedding_service",
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
        "src.docarag.services.vector_db.get_embedding_service",
        return_value=mock_embedding_client,
    ):
        with patch(
            "src.docarag.services.vector_db.is_collection_exists", return_value=True
        ):
            with pytest.raises(Exception, match="Embedding service unavailable"):
                await find_nearest_vectors(
                    query="test query", collection_name="TestCollection", limit=10
                )


@pytest.mark.asyncio
async def test_delete_objects_by_document_name_filters_by_document(
    mock_weaviate_client,
):
    """Test that only the chunks of the requested document are deleted."""
    mock_collection = Mock()
    mock_collection.data = Mock()
    mock_collection.data.delete_many = AsyncMock(
        return_value=DeleteManyReturn(failed=0, matches=4, objects=None, successful=4)
    )
    mock_weaviate_client.collections.use = Mock(return_value=mock_collection)

    with (
        patch(
            "src.docarag.services.vector_db.get_vector_db_client",
            return_value=mock_weaviate_client,
        ),
        patch("src.docarag.services.vector_db.is_collection_exists", return_value=True),
    ):
        deleted = await delete_objects_by_document_name(
            "TestCollection", "test_doc.pdf"
        )

    assert deleted == 4
    mock_collection.data.delete_many.assert_awaited_once()
    where_filter = mock_collection.data.delete_many.call_args.kwargs["where"]
    assert where_filter.target == "document_name"
    assert where_filter.value == "test_doc.pdf"


@pytest.mark.asyncio
async def test_delete_objects_by_document_name_missing_collection_returns_zero():
    """Test that purging a collection that does not exist is a no-op."""
    with patch(
        "src.docarag.services.vector_db.is_collection_exists", return_value=False
    ):
        deleted = await delete_objects_by_document_name(
            "NonExistentCollection", "test_doc.pdf"
        )

    assert deleted == 0


@pytest.fixture
def batch_objects():
    return [
        {
            "properties": {"document_name": "a.md", "page": i, "content": f"c{i}"},
            "vector": {"content_vector": [0.1, 0.2]},
        }
        for i in range(3)
    ]


@pytest.mark.asyncio
async def test_add_batch_objects_missing_collection_raises(batch_objects):
    """Test that inserting into a missing collection fails loudly instead of no-op."""
    with patch(
        "src.docarag.services.vector_db.is_collection_exists", return_value=False
    ):
        with pytest.raises(ValueError, match="does not exist"):
            await add_batch_objects("Missing", batch_objects)


@pytest.mark.asyncio
async def test_add_batch_objects_uses_insert_many(mock_weaviate_client, batch_objects):
    """Test that all objects go through one insert_many call per batch."""
    mock_collection = Mock()
    result = Mock()
    result.has_errors = False
    mock_collection.data.insert_many = AsyncMock(return_value=result)
    mock_weaviate_client.collections.get = Mock(return_value=mock_collection)

    with (
        patch(
            "src.docarag.services.vector_db.get_vector_db_client",
            return_value=mock_weaviate_client,
        ),
        patch("src.docarag.services.vector_db.is_collection_exists", return_value=True),
    ):
        await add_batch_objects("TestCollection", batch_objects)

    mock_collection.data.insert_many.assert_awaited_once()
    objects = mock_collection.data.insert_many.call_args.args[0]
    assert len(objects) == 3
    assert objects[0].properties["document_name"] == "a.md"
    assert objects[0].vector == {"content_vector": [0.1, 0.2]}


@pytest.mark.asyncio
async def test_add_batch_objects_insert_errors_raise(
    mock_weaviate_client, batch_objects
):
    """Test that a rejected object surfaces as an exception, not a warning."""
    mock_collection = Mock()
    error = Mock()
    error.message = "vector lengths don't match"
    result = Mock()
    result.has_errors = True
    result.errors = {0: error}
    mock_collection.data.insert_many = AsyncMock(return_value=result)
    mock_weaviate_client.collections.get = Mock(return_value=mock_collection)

    with (
        patch(
            "src.docarag.services.vector_db.get_vector_db_client",
            return_value=mock_weaviate_client,
        ),
        patch("src.docarag.services.vector_db.is_collection_exists", return_value=True),
    ):
        with pytest.raises(WeaviateInsertManyAllFailedError, match="vector lengths"):
            await add_batch_objects("TestCollection", batch_objects)


@pytest.mark.asyncio
async def test_verify_embedding_dimension_mismatch_raises(mock_weaviate_client):
    """Test that stored vectors of another size fail startup with a clear message."""
    service = Mock()
    service.get_embedding_dimension_async = AsyncMock(return_value=1024)
    stored = Mock()
    stored.vector = {"content_vector": [0.0] * 256}
    response = Mock()
    response.objects = [stored]
    mock_collection = Mock()
    mock_collection.query.fetch_objects = AsyncMock(return_value=response)
    mock_weaviate_client.collections.get = Mock(return_value=mock_collection)

    with (
        patch(
            "src.docarag.services.vector_db.get_embedding_service", return_value=service
        ),
        patch(
            "src.docarag.services.vector_db.get_vector_db_client",
            return_value=mock_weaviate_client,
        ),
        patch("src.docarag.services.vector_db.is_collection_exists", return_value=True),
    ):
        with pytest.raises(RuntimeError, match="dimension 256 .* produces 1024"):
            await verify_embedding_dimension()


@pytest.mark.asyncio
async def test_verify_embedding_dimension_probe_failure_only_warns(caplog):
    """Test that an unreachable embedding endpoint does not block startup."""
    service = Mock()
    service.get_embedding_dimension_async = AsyncMock(
        side_effect=EmbeddingError("connection refused")
    )

    with (
        patch(
            "src.docarag.services.vector_db.get_embedding_service", return_value=service
        ),
        caplog.at_level("WARNING", logger="src.docarag.services.vector_db"),
    ):
        assert await verify_embedding_dimension() is None

    assert any("skipping dimension check" in r.message for r in caplog.records)


def _status_error(status: int, body: dict):
    import httpx

    return UnexpectedStatusCodeError(
        "Collection may not have been created properly.",
        httpx.Response(
            status, json=body, request=httpx.Request("POST", "http://w/v1/schema")
        ),
    )


@pytest.mark.asyncio
async def test_create_default_collection_tolerates_a_concurrent_create(
    mock_weaviate_client,
):
    """Two uvicorn workers race at startup: the loser's 422 'already exists' is not an error."""
    mock_weaviate_client.collections.exists = AsyncMock(return_value=False)
    mock_weaviate_client.collections.create = AsyncMock(
        side_effect=_status_error(
            422, {"error": [{"message": "TYPE_ADD_CLASS: class already exists"}]}
        )
    )
    with patch(
        "src.docarag.services.vector_db.get_vector_db_client",
        return_value=mock_weaviate_client,
    ):
        await create_default_collection()

    mock_weaviate_client.collections.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_default_collection_reraises_other_schema_errors(
    mock_weaviate_client,
):
    mock_weaviate_client.collections.exists = AsyncMock(return_value=False)
    mock_weaviate_client.collections.create = AsyncMock(
        side_effect=_status_error(
            422, {"error": [{"message": "invalid property name"}]}
        )
    )
    with patch(
        "src.docarag.services.vector_db.get_vector_db_client",
        return_value=mock_weaviate_client,
    ):
        with pytest.raises(UnexpectedStatusCodeError, match="invalid property name"):
            await create_default_collection()
