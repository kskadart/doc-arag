import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


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


# --- sessions --------------------------------------------------------------------


def _fake_query_documents(captured):
    from src.docarag.models.responses import AgentQueryResponse

    async def fake(request):
        captured["session_id"] = request.session_id
        return AgentQueryResponse(
            query=request.query,
            answer="ok",
            rephrased_query=None,
            confidence=0.9,
            iterations=1,
            sources_used=0,
            session_id=request.session_id,
        )

    return fake


def test_query_forwards_and_echoes_session_id(client):
    test_client = client[0]
    captured = {}

    with patch(
        "src.docarag.services.agent.query_documents",
        side_effect=_fake_query_documents(captured),
    ):
        response = test_client.post(
            "/query", json={"query": "q", "session_id": "m2k9x1abc-42"}
        )

    assert response.status_code == 200
    assert captured["session_id"] == "m2k9x1abc-42"
    assert response.json()["session_id"] == "m2k9x1abc-42"


def test_query_blank_session_id_means_stateless(client):
    test_client = client[0]
    captured = {}

    with patch(
        "src.docarag.services.agent.query_documents",
        side_effect=_fake_query_documents(captured),
    ):
        response = test_client.post("/query", json={"query": "q", "session_id": "  "})

    assert response.status_code == 200
    assert captured["session_id"] is None
    assert response.json()["session_id"] is None


@pytest.mark.parametrize("bad_id", ["a b", "../x", "x" * 65, "чат", "-leading"])
def test_query_rejects_malformed_session_id(client, bad_id):
    test_client = client[0]
    response = test_client.post("/query", json={"query": "q", "session_id": bad_id})
    assert response.status_code == 422


def test_get_session_history_returns_messages_and_summary(client):
    from datetime import UTC, datetime

    from src.docarag.models.sessions import ChatTurn, StoredSession

    test_client = client[0]
    now = datetime(2026, 9, 14, tzinfo=UTC)
    stored = StoredSession(
        session_id="chat-1",
        messages=[
            ChatTurn(role="user", content="q", created_at=now, turn_index=0),
            ChatTurn(
                role="assistant",
                content="a",
                created_at=now,
                turn_index=1,
                standalone_query="q resolved",
                confidence=0.8,
                source_documents=["a.md"],
                source_domains=["elan"],
            ),
        ],
        summary="earlier",
        summary_covers_messages=0,
        created_at=now,
        updated_at=now,
    )
    store = AsyncMock()
    store.load.return_value = stored

    with patch("src.docarag.api.get_session_store", return_value=store):
        response = test_client.get("/sessions/chat-1")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "chat-1"
    assert data["message_count"] == 2
    assert data["summary"] == "earlier"
    assert data["messages"][1]["standalone_query"] == "q resolved"
    assert data["messages"][1]["source_documents"] == ["a.md"]
    store.load.assert_awaited_once()
    assert store.load.call_args.args[0] == "chat-1"


def test_get_session_history_unknown_is_404_and_invalid_is_422(client):
    test_client = client[0]
    store = AsyncMock()
    store.load.return_value = None

    with patch("src.docarag.api.get_session_store", return_value=store):
        assert test_client.get("/sessions/nope").status_code == 404
        assert test_client.get("/sessions/bad%20id").status_code == 422


def test_session_store_failure_is_503(client):
    from src.docarag.errors import SessionStoreError

    test_client = client[0]
    store = AsyncMock()
    store.load.side_effect = SessionStoreError("down")
    store.delete.side_effect = SessionStoreError("down")

    with patch("src.docarag.api.get_session_store", return_value=store):
        assert test_client.get("/sessions/chat-1").status_code == 503
        assert test_client.delete("/sessions/chat-1").status_code == 503


def test_delete_session_is_idempotent(client):
    test_client = client[0]
    store = AsyncMock()
    store.delete.return_value = 4

    with patch("src.docarag.api.get_session_store", return_value=store):
        first = test_client.delete("/sessions/chat-1")
        store.delete.return_value = 0
        second = test_client.delete("/sessions/chat-1")

    assert first.status_code == 200
    assert first.json() == {
        "session_id": "chat-1",
        "status": "deleted",
        "deleted_messages": 4,
    }
    assert second.status_code == 200 and second.json()["deleted_messages"] == 0


def test_lifespan_prepares_sessions_and_stops_cleanup_loop(monkeypatch):
    """The fixture skips the lifespan, so exercise it explicitly."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        "src.docarag.settings.settings.session_cleanup_interval_minutes", 60
    )
    with (
        patch("src.docarag.api.check_vector_db_connection", AsyncMock()),
        patch("src.docarag.api.create_default_collection", AsyncMock()),
        patch("src.docarag.api.create_session_collection", AsyncMock()) as create,
        patch("src.docarag.api.sweep_expired_sessions", AsyncMock()) as sweep,
        patch("src.docarag.api.verify_embedding_dimension", AsyncMock()),
    ):
        from src.docarag.api import app as app_instance

        with TestClient(app_instance) as test_client:
            assert test_client.get("/health").status_code == 200

    create.assert_awaited_once()
    sweep.assert_awaited_once()
