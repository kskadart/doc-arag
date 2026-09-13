import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client():
    """Create a test client."""
    with (
        patch("src.docarag.api.check_vector_db_connection"),
        patch("src.docarag.api.create_default_collection"),
    ):
        from src.docarag.api import app as app_instance
        from src.docarag.dependencies import get_all_files

        return TestClient(app_instance), app_instance, get_all_files


def test_health_check(client):
    """Test health check endpoint."""
    test_client = client[0]
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_upload_document_invalid_type(client):
    """Test uploading document with invalid file type."""
    test_client = client[0]
    files = {"file": ("test.txt", b"test content", "text/plain")}
    response = test_client.post("/uploads", files=files)
    assert response.status_code == 422


def test_query_empty(client):
    """Test query with empty text."""
    test_client = client[0]
    response = test_client.post("/query", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.skip(reason="NOT IMPLEMENTED - endpoint implementation is commented out")
def test_query_with_valid_request(client):
    """Test query with valid request."""
    test_client = client[0]
    response = test_client.post("/query", json={"query": "test question"})
    assert response.status_code == 200


@pytest.mark.skip(reason="NOT IMPLEMENTED - endpoint implementation is commented out")
def test_get_task_status_not_found(client):
    """Test getting status of non-existent task."""
    test_client = client[0]
    response = test_client.get("/tasks/nonexistent-task-id")
    assert response.status_code == 404


@patch("src.docarag.api.delete_file_by_id")
@patch("src.docarag.api.get_minio_client")
def test_delete_document_not_found(mock_get_minio, mock_delete, client):
    """Test deleting non-existent document."""
    test_client, app_instance, get_all_files_orig = client
    app_instance.dependency_overrides[get_all_files_orig] = lambda: []
    mock_get_minio.return_value = None

    response = test_client.delete("/documents/nonexistent-id")
    assert response.status_code == 404

    app_instance.dependency_overrides.clear()


def test_query_legacy_collection_name_means_no_domain_filter(client):
    """Test that the client's hardcoded collection name is not used as a filter."""
    from src.docarag.models.responses import AgentQueryResponse

    test_client = client[0]
    captured = {}

    async def fake_query_documents(request):
        captured["domain"] = request.domain
        return AgentQueryResponse(
            query=request.query,
            answer="ok",
            rephrased_query=None,
            confidence=0.9,
            iterations=1,
            sources_used=1,
        )

    with patch(
        "src.docarag.services.agent.query_documents", side_effect=fake_query_documents
    ):
        response = test_client.post(
            "/query", json={"query": "q", "domain": "DefaultDocuments"}
        )

    assert response.status_code == 200
    assert captured["domain"] is None


def test_query_domain_reaches_agent(client):
    """Test that a real domain slug is forwarded to the agent."""
    from src.docarag.models.responses import AgentQueryResponse

    test_client = client[0]
    captured = {}

    async def fake_query_documents(request):
        captured["domain"] = request.domain
        return AgentQueryResponse(
            query=request.query,
            answer="ok",
            rephrased_query=None,
            confidence=0.9,
            iterations=1,
            sources_used=1,
        )

    with patch(
        "src.docarag.services.agent.query_documents", side_effect=fake_query_documents
    ):
        response = test_client.post(
            "/query", json={"query": "q", "domain": "diagnostics"}
        )

    assert response.status_code == 200
    assert captured["domain"] == "diagnostics"


def test_health_reports_model_configuration(client):
    """Test that /health exposes the configured providers for eval reports."""
    test_client = client[0]
    data = test_client.get("/health").json()
    assert data["llm_provider"] == "openai"
    assert data["llm_model"] == "test/chat-model"
    assert data["embedding_model"] == "test/embedding-model"
    assert data["reranker_provider"] == "grpc"  # conftest pins the gRPC provider
