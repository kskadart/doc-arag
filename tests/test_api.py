import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.docarag.api import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_upload_document_invalid_type(client):
    """Test uploading document with invalid file type."""
    files = {"file": ("test.txt", b"test content", "text/plain")}
    response = client.post("/uploads", files=files)
    assert response.status_code == 422


def test_query_empty(client):
    """Test query with empty text."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.skip(reason="NOT IMPLEMENTED - endpoint implementation is commented out")
def test_query_with_valid_request(client):
    """Test query with valid request."""
    response = client.post("/query", json={"query": "test question"})
    assert response.status_code == 200


@pytest.mark.skip(reason="NOT IMPLEMENTED - endpoint implementation is commented out")
def test_get_task_status_not_found(client):
    """Test getting status of non-existent task."""
    response = client.get("/tasks/nonexistent-task-id")
    assert response.status_code == 404


def test_delete_document_not_found(client):
    """Test deleting non-existent document."""
    with patch("src.docarag.api.get_minio_client"):
        with patch("src.docarag.api.list_all_files") as mock_list:
            mock_list.return_value = []

            response = client.delete("/documents/nonexistent-id")
            assert response.status_code == 404
