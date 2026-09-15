"""Identity from the edge proxy and group-based authorization."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from unittest.mock import patch

from src.docarag.settings import Settings, settings

SECRET = "edge-shared-secret"


@pytest.fixture
def client():
    with (
        patch("src.docarag.api.check_vector_db_connection"),
        patch("src.docarag.api.create_default_collection"),
    ):
        from src.docarag.api import app as app_instance
        from src.docarag.dependencies import get_all_files

        app_instance.dependency_overrides[get_all_files] = lambda: []
        yield TestClient(app_instance)
        app_instance.dependency_overrides.clear()


@pytest.fixture
def trusted_headers(monkeypatch):
    monkeypatch.setattr(settings, "auth_trusted_headers", True)
    monkeypatch.setattr(settings, "auth_proxy_secret", SecretStr(SECRET))


PROXY = {"X-Auth-Proxy-Secret": SECRET}
OPERATOR = {**PROXY, "Remote-User": "olga", "Remote-Groups": "operators"}
ADMIN = {
    **PROXY,
    "Remote-User": "kirill",
    "Remote-Groups": "admins, operators",
    "Remote-Name": "Kirill S",
    "Remote-Email": "kirill@example.com",
}


def test_login_disabled_is_anonymous_admin_without_fake_groups(client):
    """Local runs and tests: nobody logs in, everything is allowed."""
    data = client.get("/me").json()
    assert data == {
        "username": "anonymous",
        "display_name": None,
        "email": None,
        "groups": [],
        "is_admin": True,
        "auth_mode": "none",
    }


def test_login_disabled_ignores_headers(client):
    assert client.get("/me", headers=OPERATOR).json()["username"] == "anonymous"


def test_trusted_headers_require_the_proxy_secret(client, trusted_headers):
    """Identity headers without the secret are a bypass attempt, not a login."""
    spoof = {"Remote-User": "mallory", "Remote-Groups": "admins"}
    assert client.get("/me", headers=spoof).status_code == 401
    assert client.get("/documents", headers=spoof).status_code == 401
    wrong = {**spoof, "X-Auth-Proxy-Secret": "guess"}
    assert client.get("/me", headers=wrong).status_code == 401


def test_trusted_headers_without_identity_is_401(client, trusted_headers):
    assert client.get("/me", headers=PROXY).status_code == 401
    assert client.get("/documents", headers=PROXY).status_code == 401
    assert client.post("/query", json={"query": "q"}, headers=PROXY).status_code == 401
    assert client.get("/tasks/x", headers=PROXY).status_code == 401


def test_health_stays_public(client, trusted_headers):
    assert client.get("/health").status_code == 200


def test_me_reports_identity(client, trusted_headers):
    assert client.get("/me", headers=ADMIN).json() == {
        "username": "kirill",
        "display_name": "Kirill S",
        "email": "kirill@example.com",
        "groups": ["admins", "operators"],
        "is_admin": True,
        "auth_mode": "trusted-headers",
    }


def test_utf8_display_name_survives_latin1_header_decoding(client, trusted_headers):
    headers = {**OPERATOR, "Remote-Name": "Ольга Оператор".encode("utf-8")}
    data = client.get("/me", headers=headers).json()
    assert data["display_name"] == "Ольга Оператор"


def test_operator_is_not_admin(client, trusted_headers):
    data = client.get("/me", headers=OPERATOR).json()
    assert data["is_admin"] is False
    assert data["groups"] == ["operators"]


def test_document_management_requires_admin_group(client, trusted_headers):
    assert client.get("/documents", headers=OPERATOR).status_code == 403
    assert client.delete("/documents/x", headers=OPERATOR).status_code == 403
    assert client.post("/embeddings/x", headers=OPERATOR).status_code == 403
    assert client.post("/uploads", headers=OPERATOR).status_code == 403
    scrape = {"url": "https://example.com"}
    assert client.post("/scrappings", json=scrape, headers=OPERATOR).status_code == 403

    # The admin passes the guard and reaches the handlers
    assert client.get("/documents", headers=ADMIN).status_code == 200
    assert client.delete("/documents/x", headers=ADMIN).status_code == 404
    assert client.post("/scrappings", json=scrape, headers=ADMIN).status_code == 501


def test_operator_may_query(client, trusted_headers):
    from src.docarag.models.responses import AgentQueryResponse

    async def fake_query_documents(request):
        return AgentQueryResponse(
            query=request.query,
            answer="ok",
            rephrased_query=None,
            confidence=0.9,
            iterations=1,
            sources_used=0,
        )

    with patch(
        "src.docarag.services.agent.query_documents", side_effect=fake_query_documents
    ):
        response = client.post("/query", json={"query": "q"}, headers=OPERATOR)
    assert response.status_code == 200


def test_admin_group_name_is_configurable(client, trusted_headers, monkeypatch):
    monkeypatch.setattr(settings, "auth_admin_group", "oreo-admins")
    assert client.get("/me", headers=ADMIN).json()["is_admin"] is False
    headers = {**PROXY, "Remote-User": "x", "Remote-Groups": "oreo-admins"}
    assert client.get("/me", headers=headers).json()["is_admin"] is True


def test_settings_refuse_trusted_headers_without_secret():
    with pytest.raises(ValidationError, match="AUTH_PROXY_SECRET"):
        Settings(auth_trusted_headers=True, auth_proxy_secret=SecretStr(""))
    assert Settings(auth_trusted_headers=True, auth_proxy_secret=SecretStr("x"))
