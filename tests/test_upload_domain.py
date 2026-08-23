from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from src.docarag.consts import DEFAULT_COLLECTION_NAME, DEFAULT_DOMAIN
from src.docarag.clients.minio_client import extract_domain
from src.docarag.models.upload import UploadModel
from src.docarag.services.uploader import upload_document
from src.docarag.tasks.embedding_task import run_embedding_task


@pytest.fixture
def parsed_chunks():
    """Chunks as produced by the document parser."""
    return [
        {"content": "first chunk of the document", "page": 1},
        {"content": "second chunk of the document", "page": 2},
    ]


def test_upload_model_accepts_slug_domain():
    """Test that a lowercase slug domain passes validation."""
    model = UploadModel(
        document_name="guide.md",
        document_url="https://example.com/guide.md",
        domain="lapa-navigation",
    )

    assert model.domain == "lapa-navigation"


def test_upload_model_defaults_domain_when_omitted():
    """Test that a document without a domain falls back to the default one."""
    model = UploadModel(
        document_name="guide.md", document_url="https://example.com/guide.md"
    )

    assert model.domain == DEFAULT_DOMAIN


@pytest.mark.parametrize(
    "invalid_domain", ["Lapa", "-lapa", "lapa navigation", "lapa_navigation", ""]
)
def test_upload_model_rejects_non_slug_domain(invalid_domain):
    """Test that a domain which is not a slug is rejected."""
    with pytest.raises(ValidationError):
        UploadModel(
            document_name="guide.md",
            document_url="https://example.com/guide.md",
            domain=invalid_domain,
        )


def test_upload_document_stores_domain_in_object_metadata():
    """Test that the domain reaches the MinIO object metadata."""
    with (
        patch("src.docarag.services.uploader.get_minio_client", return_value=Mock()),
        patch("src.docarag.services.uploader.ensure_bucket_exists"),
        patch(
            "src.docarag.services.uploader.upload_file_to_minio",
            return_value="file-id/guide.md",
        ) as mock_upload,
    ):
        result = upload_document(
            file_content=b"# Guide",
            filename="guide.md",
            file_id="file-id",
            detected_type="md",
            domain="diagnostics",
        )

    assert mock_upload.call_args.kwargs["metadata"] == {
        "type": "md",
        "domain": "diagnostics",
    }
    assert mock_upload.call_args.kwargs["content_type"] == "text/markdown"
    assert result["domain"] == "diagnostics"


def test_extract_domain_reads_prefixed_metadata_header():
    """Test that the domain is read from the S3 user metadata header."""
    assert extract_domain({"X-Amz-Meta-Domain": "billing"}) == "billing"


def test_extract_domain_falls_back_to_default_without_metadata():
    """Test that objects uploaded without a domain get the default one."""
    assert extract_domain({"X-Amz-Meta-Type": "pdf"}) == DEFAULT_DOMAIN
    assert extract_domain(None) == DEFAULT_DOMAIN


@pytest.mark.asyncio
async def test_embedding_task_stores_domain_in_chunk_properties(parsed_chunks):
    """Test that the domain of the document is written to every chunk."""
    embedding_service = Mock()
    embedding_service.embed_batch_async = AsyncMock(return_value=[[0.1], [0.2]])

    with (
        patch("src.docarag.tasks.embedding_task.get_minio_client", return_value=Mock()),
        patch(
            "src.docarag.tasks.embedding_task.download_file_by_id",
            return_value=(
                b"# Guide",
                "guide.md",
                {"content_type": "text/markdown", "domain": "diagnostics"},
            ),
        ),
        patch(
            "src.docarag.tasks.embedding_task.parse_document",
            return_value=parsed_chunks,
        ),
        patch(
            "src.docarag.tasks.embedding_task.get_embedding_service",
            return_value=embedding_service,
        ),
        patch(
            "src.docarag.tasks.embedding_task.delete_objects_by_document_name",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch(
            "src.docarag.tasks.embedding_task.add_batch_objects", new_callable=AsyncMock
        ) as mock_add,
    ):
        await run_embedding_task("task-id", "file-id")

    stored_objects = mock_add.call_args.args[1]

    assert len(stored_objects) == 2
    for stored in stored_objects:
        assert stored["properties"]["domain"] == "diagnostics"


@pytest.mark.asyncio
async def test_embedding_task_defaults_domain_for_legacy_objects(parsed_chunks):
    """Test that a document stored without a domain gets the default one."""
    embedding_service = Mock()
    embedding_service.embed_batch_async = AsyncMock(return_value=[[0.1], [0.2]])

    with (
        patch("src.docarag.tasks.embedding_task.get_minio_client", return_value=Mock()),
        patch(
            "src.docarag.tasks.embedding_task.download_file_by_id",
            return_value=(b"# Guide", "guide.md", {"content_type": "text/markdown"}),
        ),
        patch(
            "src.docarag.tasks.embedding_task.parse_document",
            return_value=parsed_chunks,
        ),
        patch(
            "src.docarag.tasks.embedding_task.get_embedding_service",
            return_value=embedding_service,
        ),
        patch(
            "src.docarag.tasks.embedding_task.delete_objects_by_document_name",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch(
            "src.docarag.tasks.embedding_task.add_batch_objects", new_callable=AsyncMock
        ) as mock_add,
    ):
        await run_embedding_task("task-id", "file-id")

    stored_objects = mock_add.call_args.args[1]

    assert stored_objects[0]["properties"]["domain"] == DEFAULT_DOMAIN


@pytest.mark.asyncio
async def test_embedding_task_purges_previous_chunks_before_insert(parsed_chunks):
    """Test that re-embedding a document removes its previous chunks first."""
    embedding_service = Mock()
    embedding_service.embed_batch_async = AsyncMock(return_value=[[0.1], [0.2]])
    call_order = []

    with (
        patch("src.docarag.tasks.embedding_task.get_minio_client", return_value=Mock()),
        patch(
            "src.docarag.tasks.embedding_task.download_file_by_id",
            return_value=(
                b"# Guide",
                "guide.md",
                {"content_type": "text/markdown", "domain": "diagnostics"},
            ),
        ),
        patch(
            "src.docarag.tasks.embedding_task.parse_document",
            return_value=parsed_chunks,
        ),
        patch(
            "src.docarag.tasks.embedding_task.get_embedding_service",
            return_value=embedding_service,
        ),
        patch(
            "src.docarag.tasks.embedding_task.delete_objects_by_document_name",
            new_callable=AsyncMock,
            side_effect=lambda *args: call_order.append("purge") or 2,
        ) as mock_purge,
        patch(
            "src.docarag.tasks.embedding_task.add_batch_objects",
            new_callable=AsyncMock,
            side_effect=lambda *args: call_order.append("insert"),
        ),
    ):
        await run_embedding_task("task-id", "file-id")

    mock_purge.assert_awaited_once_with(DEFAULT_COLLECTION_NAME, "guide.md")
    assert call_order == ["purge", "insert"]


@pytest.fixture
def upload_client():
    """Test client with the startup dependencies of the API mocked out."""
    with (
        patch("src.docarag.api.check_vector_db_connection"),
        patch("src.docarag.api.create_default_collection"),
    ):
        from fastapi.testclient import TestClient
        from src.docarag.api import app

        return TestClient(app)


def test_upload_endpoint_passes_domain_to_upload_service(upload_client):
    """Test that the domain form field reaches the upload service."""
    with patch(
        "src.docarag.api.process_upload",
        new_callable=AsyncMock,
        return_value={
            "file_id": "file-id",
            "filename": "guide.md",
            "object_key": "file-id/guide.md",
        },
    ) as mock_process:
        response = upload_client.post(
            "/uploads",
            data={"document_name": "guide.md", "domain": "diagnostics"},
            files={"document": ("guide.md", b"# Guide", "text/markdown")},
        )

    assert response.status_code == 200
    assert mock_process.await_args.args[0].domain == "diagnostics"


def test_upload_endpoint_rejects_non_slug_domain(upload_client):
    """Test that an invalid domain is refused before any storage call."""
    with patch(
        "src.docarag.api.process_upload", new_callable=AsyncMock
    ) as mock_process:
        response = upload_client.post(
            "/uploads",
            data={"document_name": "guide.md", "domain": "Diagnostics"},
            files={"document": ("guide.md", b"# Guide", "text/markdown")},
        )

    assert response.status_code == 400
    mock_process.assert_not_awaited()
