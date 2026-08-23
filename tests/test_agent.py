"""Tests for the LangGraph RAG agent."""

import logging
from unittest.mock import AsyncMock, patch

import grpc
import pytest

from src.docarag.services.agent import (
    AgentState,
    rerank_documents_node,
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

    with patch("src.docarag.services.agent.RerankerService", return_value=mock_service):
        result = await rerank_documents_node(state)

    assert result == {"retrieved_docs": reranked_docs}
    mock_service.rerank_async.assert_called_once()


@pytest.mark.asyncio
async def test_rerank_documents_node_reranker_unavailable_falls_back_and_warns(
    caplog,
):
    """Test that a gRPC failure falls back to the retrieval order and logs a WARNING."""
    retrieved_docs = [
        {"content": f"doc {i}", "document_name": f"{i}.md"}
        for i in range(settings.rerank_top_k + 2)
    ]
    state = AgentState(query="test query", retrieved_docs=retrieved_docs)

    mock_service = AsyncMock()
    mock_service.rerank_async.side_effect = grpc.RpcError("reranker unreachable")

    with (
        patch("src.docarag.services.agent.RerankerService", return_value=mock_service),
        caplog.at_level(logging.WARNING, logger="src.docarag.services.agent"),
    ):
        result = await rerank_documents_node(state)

    assert result == {"retrieved_docs": retrieved_docs[: settings.rerank_top_k]}
    assert any(
        "Reranker service unavailable" in record.message for record in caplog.records
    )
