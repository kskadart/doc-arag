from datetime import datetime
from unittest.mock import Mock, patch
import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """
    Set environment variables for testing.
    """
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_MODEL"] = "test-model"
    os.environ["MINIO_ENDPOINT"] = "localhost:9000"
    os.environ["MINIO_ACCESS_KEY"] = "test-access"
    os.environ["MINIO_SECRET_KEY"] = "test-secret"
    os.environ["MINIO_BUCKET"] = "test-bucket"
    os.environ["MINIO_SECURE"] = "false"


@pytest.fixture
def client():
    """
    Create test client with mocked dependencies.
    """
    with patch("src.docarag.api.check_vector_db_connection"), \
         patch("src.docarag.api.create_default_collection"):
        from src.docarag.api import app
        return TestClient(app)


@pytest.fixture
def mock_minio_files():
    """
    Mock MinIO files data.
    """
    return [
        {
            "file_id": "test-file-id-1",
            "object_key": "test-file-id-1/document1.pdf",
            "filename": "document1.pdf",
            "size_bytes": 1024,
            "content_type": "application/pdf",
            "last_modified": datetime(2025, 1, 1, 12, 0, 0),
            "metadata": {"type": "pdf", "filename": "document1.pdf"},
        },
        {
            "file_id": "test-file-id-2",
            "object_key": "test-file-id-2/document2.docx",
            "filename": "document2.docx",
            "size_bytes": 2048,
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "last_modified": datetime(2025, 1, 2, 12, 0, 0),
            "metadata": {"type": "docx", "filename": "document2.docx"},
        },
    ]


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
def test_list_uploaded_files_success(mock_list_files, mock_get_client, mock_minio_files, client):
    """
    Test successful listing of uploaded files.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = mock_minio_files

    response = client.get("/documents?page=1&page_size=10")

    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 2
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["files"]) == 2
    
    assert data["files"][0]["file_id"] == "test-file-id-1"
    assert data["files"][0]["filename"] == "document1.pdf"
    assert data["files"][0]["size_bytes"] == 1024
    assert data["files"][0]["content_type"] == "application/pdf"
    
    assert data["files"][1]["file_id"] == "test-file-id-2"
    assert data["files"][1]["filename"] == "document2.docx"


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
def test_list_uploaded_files_pagination(mock_list_files, mock_get_client, mock_minio_files, client):
    """
    Test pagination of uploaded files list.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = mock_minio_files

    response = client.get("/documents?page=1&page_size=1")

    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 2
    assert data["page"] == 1
    assert data["page_size"] == 1
    assert len(data["files"]) == 1
    assert data["files"][0]["file_id"] == "test-file-id-1"


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
def test_list_uploaded_files_second_page(mock_list_files, mock_get_client, mock_minio_files, client):
    """
    Test second page of uploaded files list.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = mock_minio_files

    response = client.get("/documents?page=2&page_size=1")

    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 2
    assert data["page"] == 2
    assert data["page_size"] == 1
    assert len(data["files"]) == 1
    assert data["files"][0]["file_id"] == "test-file-id-2"


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
def test_list_uploaded_files_empty(mock_list_files, mock_get_client, client):
    """
    Test listing when no files are uploaded.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = []

    response = client.get("/documents?page=1&page_size=10")

    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 0
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["files"]) == 0


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
def test_list_uploaded_files_error(mock_list_files, mock_get_client, client):
    """
    Test error handling when listing files fails.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.side_effect = Exception("MinIO connection error")

    response = client.get("/documents?page=1&page_size=10")

    assert response.status_code == 500
    assert "Error listing uploaded files" in response.json()["detail"]


def test_list_uploaded_files_invalid_page(client):
    """
    Test validation for invalid page number.
    """
    response = client.get("/documents?page=0&page_size=10")
    
    assert response.status_code == 422


def test_list_uploaded_files_invalid_page_size(client):
    """
    Test validation for invalid page size.
    """
    response = client.get("/documents?page=1&page_size=101")
    
    assert response.status_code == 422


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
def test_list_uploaded_files_metadata_included(mock_list_files, mock_get_client, mock_minio_files, client):
    """
    Test that metadata is properly included in response.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = mock_minio_files

    response = client.get("/documents?page=1&page_size=10")

    assert response.status_code == 200
    data = response.json()
    
    assert "metadata" in data["files"][0]
    assert data["files"][0]["metadata"]["type"] == "pdf"
    assert data["files"][0]["metadata"]["filename"] == "document1.pdf"


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
@patch("src.docarag.api.delete_file_by_id")
def test_delete_uploaded_file_success(mock_delete_file, mock_list_files, mock_get_client, mock_minio_files, client):
    """
    Test successful deletion of an uploaded file.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = mock_minio_files
    mock_delete_file.return_value = 1

    response = client.delete("/documents/test-file-id-1")

    assert response.status_code == 200
    data = response.json()
    
    assert data["file_id"] == "test-file-id-1"
    assert data["status"] == "deleted"
    assert "Successfully deleted 1 file(s)" in data["message"]
    
    mock_delete_file.assert_called_once()


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
def test_delete_uploaded_file_not_found(mock_list_files, mock_get_client, client):
    """
    Test deletion of a non-existent file.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = []

    response = client.delete("/documents/non-existent-id")

    assert response.status_code == 404
    assert "No file found with ID" in response.json()["detail"]


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
@patch("src.docarag.api.delete_file_by_id")
def test_delete_uploaded_file_multiple_objects(mock_delete_file, mock_list_files, mock_get_client, mock_minio_files, client):
    """
    Test deletion when multiple objects are associated with a file_id.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = mock_minio_files
    mock_delete_file.return_value = 3

    response = client.delete("/documents/test-file-id-1")

    assert response.status_code == 200
    data = response.json()
    
    assert data["file_id"] == "test-file-id-1"
    assert data["status"] == "deleted"
    assert "Successfully deleted 3 file(s)" in data["message"]


@patch("src.docarag.api.get_minio_client")
@patch("src.docarag.api.list_all_files")
@patch("src.docarag.api.delete_file_by_id")
def test_delete_uploaded_file_error(mock_delete_file, mock_list_files, mock_get_client, mock_minio_files, client):
    """
    Test error handling when deletion fails.
    """
    mock_get_client.return_value = Mock()
    mock_list_files.return_value = mock_minio_files
    mock_delete_file.side_effect = Exception("MinIO deletion error")

    response = client.delete("/documents/test-file-id-1")

    assert response.status_code == 500
    assert "Error deleting uploaded file" in response.json()["detail"]

