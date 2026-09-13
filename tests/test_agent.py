"""Tests for the LangGraph RAG agent."""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.docarag.errors import RerankerError
from src.docarag.services.agent import (
    AgentState,
    parse_confidence,
    rerank_documents_node,
    retrieve_documents_node,
    should_continue,
)
from src.docarag.settings import settings


def test_should_continue():
    """Test conditional routing logic based on agent state."""
    state_continue = AgentState(
        query="test",
        confidence=0.5,
        iterations=1,
        should_iterate=True,
        max_iterations=2,
    )

    assert should_continue(state_continue) == "rephrase_query"

    state_end = AgentState(
        query="test",
        confidence=0.9,
        iterations=1,
        should_iterate=False,
        max_iterations=2,
    )

    assert should_continue(state_end) == "end"


def test_parse_confidence_extracts_first_number_and_clamps():
    """Test that chatty evaluator replies still yield a usable score."""
    assert parse_confidence("0.85") == 0.85
    assert parse_confidence("Confidence Score: 0.7\nBecause...") == 0.7
    assert parse_confidence("I'd say 1.5") == 1.0
    assert parse_confidence("no number here") is None


@pytest.mark.asyncio
async def test_rerank_documents_node_reranker_available_returns_reranked_docs():
    """Test that the node returns the reranker service's reordered documents."""
    retrieved_docs = [
        {"content": "low relevance", "document_name": "a.md"},
        {"content": "high relevance", "document_name": "b.md"},
    ]
    reranked_docs = [
        {"content": "high relevance", "document_name": "b.md", "rerank_score": 0.9},
        {"content": "low relevance", "document_name": "a.md", "rerank_score": 0.1},
    ]
    state = AgentState(query="test query", retrieved_docs=retrieved_docs)

    mock_service = AsyncMock()
    mock_service.rerank_async.return_value = reranked_docs

    with patch(
        "src.docarag.services.agent.get_reranker_service", return_value=mock_service
    ):
        result = await rerank_documents_node(state)

    assert result == {"retrieved_docs": reranked_docs}
    mock_service.rerank_async.assert_called_once()


@pytest.mark.asyncio
async def test_rerank_documents_node_reranker_unavailable_falls_back_and_warns(
    caplog,
):
    """Test that a reranker failure falls back to the retrieval order and logs a WARNING."""
    retrieved_docs = [
        {"content": f"doc {i}", "document_name": f"{i}.md"}
        for i in range(settings.rerank_top_k + 2)
    ]
    state = AgentState(query="test query", retrieved_docs=retrieved_docs)

    mock_service = AsyncMock()
    mock_service.rerank_async.side_effect = RerankerError("reranker unreachable")

    with (
        patch(
            "src.docarag.services.agent.get_reranker_service", return_value=mock_service
        ),
        caplog.at_level(logging.WARNING, logger="src.docarag.services.agent"),
    ):
        result = await rerank_documents_node(state)

    assert result == {"retrieved_docs": retrieved_docs[: settings.rerank_top_k]}
    assert any(
        "Reranker service unavailable" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_rerank_documents_node_disabled_provider_slices_without_service(
    monkeypatch,
):
    """Test that reranker_provider=none never touches the service."""
    monkeypatch.setattr(settings, "reranker_provider", "none")
    retrieved_docs = [{"content": f"doc {i}"} for i in range(settings.rerank_top_k + 3)]
    state = AgentState(query="test query", retrieved_docs=retrieved_docs)

    with patch("src.docarag.services.agent.get_reranker_service") as get_service:
        result = await rerank_documents_node(state)

    get_service.assert_not_called()
    assert result == {"retrieved_docs": retrieved_docs[: settings.rerank_top_k]}


def _weaviate_client_with(near_vector_mock):
    collection = Mock()
    collection.query.near_vector = near_vector_mock
    client = Mock()
    client.collections.get = Mock(return_value=collection)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _object(distance):
    obj = Mock()
    obj.uuid = "uuid-1"
    obj.properties = {
        "content": "text",
        "document_name": "a.md",
        "page": 1,
        "domain": "diagnostics",
    }
    obj.metadata = Mock()
    obj.metadata.distance = distance
    return obj


@pytest.mark.asyncio
async def test_retrieve_documents_node_targets_named_vector_without_filter():
    """Test that retrieval addresses the named vector and passes no filter by default."""
    response = Mock()
    response.objects = [_object(0.25)]
    near_vector = AsyncMock(return_value=response)
    state = AgentState(query="q", query_embedding=[0.1, 0.2])

    with patch(
        "src.docarag.services.agent.get_vector_db_client",
        return_value=_weaviate_client_with(near_vector),
    ):
        result = await retrieve_documents_node(state)

    kwargs = near_vector.call_args.kwargs
    assert kwargs["target_vector"] == "content_vector"
    assert kwargs["filters"] is None
    assert kwargs["limit"] == settings.initial_retrieval_k
    assert result["retrieved_docs"][0]["similarity_score"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_retrieve_documents_node_perfect_match_scores_one():
    """Test that distance 0.0 yields similarity 1.0 (regression for the truthiness bug)."""
    response = Mock()
    response.objects = [_object(0.0)]
    near_vector = AsyncMock(return_value=response)
    state = AgentState(query="q", query_embedding=[0.1, 0.2])

    with patch(
        "src.docarag.services.agent.get_vector_db_client",
        return_value=_weaviate_client_with(near_vector),
    ):
        result = await retrieve_documents_node(state)

    assert result["retrieved_docs"][0]["similarity_score"] == 1.0


@pytest.mark.asyncio
async def test_retrieve_documents_node_applies_domain_filter():
    """Test that a domain in the state becomes a property filter."""
    response = Mock()
    response.objects = []
    near_vector = AsyncMock(return_value=response)
    state = AgentState(query="q", query_embedding=[0.1], domain="diagnostics")

    with patch(
        "src.docarag.services.agent.get_vector_db_client",
        return_value=_weaviate_client_with(near_vector),
    ):
        await retrieve_documents_node(state)

    domain_filter = near_vector.call_args.kwargs["filters"]
    assert domain_filter.target == "domain"
    assert domain_filter.value == "diagnostics"


def test_build_source_chunk_prefers_rerank_score_and_truncates():
    """Test the API-facing projection of a retrieved chunk."""
    from src.docarag.services.agent import build_source_chunk

    chunk = build_source_chunk(
        {
            "document_name": "a.md",
            "domain": "diagnostics",
            "page": 3,
            "similarity_score": 0.4,
            "rerank_score": 0.9,
            "content": "x" * 500,
        }
    )

    assert chunk.document_name == "a.md"
    assert chunk.domain == "diagnostics"
    assert chunk.page == 3
    assert chunk.score == 0.9
    assert len(chunk.snippet) == 200

    plain = build_source_chunk({"similarity_score": 0.4, "content": "short"})
    assert plain.score == 0.4 and plain.snippet == "short" and plain.page == 0
