"""Tests for the LangGraph RAG agent."""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.docarag.errors import RerankerError
from src.docarag.services.agent import (
    AgentState,
    parse_confidence,
    rerank_documents_node,
    embed_query_node,
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
async def test_retrieve_documents_node_unions_both_query_vectors():
    """Test that original and rephrased vectors are both searched and merged by uuid."""
    shared = _object(0.4)
    only_original = _object(0.2)
    only_original.uuid = "uuid-2"
    shared_better = _object(0.1)
    first, second = Mock(), Mock()
    first.objects = [shared]
    second.objects = [only_original, shared_better]
    near_vector = AsyncMock(side_effect=[first, second])
    state = AgentState(query="q", query_embedding=[0.1], original_query_embedding=[0.2])

    with patch(
        "src.docarag.services.agent.get_vector_db_client",
        return_value=_weaviate_client_with(near_vector),
    ):
        result = await retrieve_documents_node(state)

    assert near_vector.call_count == 2
    docs = result["retrieved_docs"]
    assert [doc["uuid"] for doc in docs] == ["uuid-1", "uuid-2"]
    assert docs[0]["similarity_score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_embed_query_node_embeds_original_query_when_rephrased_differs():
    """Test that the original query gets its own vector alongside the rephrased one."""
    service = Mock()
    service.embed_text_async = AsyncMock(side_effect=[[0.1], [0.2]])
    state = AgentState(query="original", rephrased_query="rephrased")

    with patch(
        "src.docarag.services.agent.get_embedding_service", return_value=service
    ):
        result = await embed_query_node(state)

    assert result == {"query_embedding": [0.1], "original_query_embedding": [0.2]}
    embedded = [call.args[0] for call in service.embed_text_async.call_args_list]
    assert embedded == ["rephrased", "original"]


@pytest.mark.asyncio
async def test_embed_query_node_skips_original_when_disabled(monkeypatch):
    """Test that the original query is not embedded when the setting is off."""
    monkeypatch.setattr(settings, "retrieval_use_original_query", False)
    service = Mock()
    service.embed_text_async = AsyncMock(return_value=[0.1])
    state = AgentState(query="original", rephrased_query="rephrased")

    with patch(
        "src.docarag.services.agent.get_embedding_service", return_value=service
    ):
        result = await embed_query_node(state)

    assert result == {"query_embedding": [0.1], "original_query_embedding": None}
    assert service.embed_text_async.call_count == 1


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


# --- conversational memory ------------------------------------------------------


def _llm_returning(text: str) -> Mock:
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=Mock(text=text))
    return llm


def _history() -> list:
    from datetime import UTC, datetime

    from src.docarag.models.sessions import ChatTurn

    now = datetime(2026, 9, 14, tzinfo=UTC)
    return [
        ChatTurn(role="user", content="Что такое ELAN?", created_at=now, turn_index=0),
        ChatTurn(
            role="assistant",
            content="ELAN — это ...",
            created_at=now,
            turn_index=1,
        ),
    ]


@pytest.mark.asyncio
async def test_rephrase_query_node_condenses_with_history_and_summary():
    """Test that the condense prompt carries the transcript, summary and latest message."""
    from src.docarag.services.agent import rephrase_query_node

    state = AgentState(
        query="А как его подключить?",
        history=_history(),
        history_summary="Ранее обсуждали тарифы.",
    )
    llm = _llm_returning("  Подключение ELAN  ")

    with patch("src.docarag.services.agent.get_chat_model", return_value=llm):
        result = await rephrase_query_node(state)

    assert result["rephrased_query"] == "Подключение ELAN"
    assert result["previous_queries"] == ["Подключение ELAN"]
    system, human = llm.ainvoke.call_args.args[0]
    assert "standalone search query" in system.content
    assert "Operator: Что такое ELAN?" in human.content
    assert "Assistant: ELAN — это ..." in human.content
    assert "Ранее обсуждали тарифы." in human.content
    assert human.content.rstrip().endswith(
        "А как его подключить?\n\nStandalone search query:"
    )


@pytest.mark.asyncio
async def test_rephrase_query_node_retry_asks_for_a_different_formulation():
    """Regression: a second iteration must not repeat the first standalone query."""
    from src.docarag.services.agent import rephrase_query_node

    state = AgentState(query="q", iterations=1, previous_queries=["first attempt"])
    llm = _llm_returning("second attempt")

    with patch("src.docarag.services.agent.get_chat_model", return_value=llm):
        result = await rephrase_query_node(state)

    human = llm.ainvoke.call_args.args[0][1]
    assert "1. first attempt" in human.content
    assert "DIFFERENT formulation" in human.content
    assert result["previous_queries"] == ["first attempt", "second attempt"]


@pytest.mark.asyncio
async def test_rephrase_query_node_without_history_says_none():
    """Test that a stateless query still goes through the condense prompt cleanly."""
    from src.docarag.services.agent import rephrase_query_node

    llm = _llm_returning("")

    with patch("src.docarag.services.agent.get_chat_model", return_value=llm):
        result = await rephrase_query_node(AgentState(query="plain question"))

    human = llm.ainvoke.call_args.args[0][1]
    assert "Recent conversation:\n(none)" in human.content
    assert result["rephrased_query"] == "plain question"  # empty reply falls back


@pytest.mark.asyncio
async def test_generate_answer_node_sends_history_as_messages(monkeypatch):
    """Test the message layout: system (rules + summary), history, context + question."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from src.docarag.services.agent import generate_answer_node

    monkeypatch.setattr(settings, "session_message_max_chars", 8)
    history = _history()
    history[1].content = "x" * 50
    state = AgentState(
        query="А как его подключить?",
        history=history,
        history_summary="Сводка.",
        retrieved_docs=[{"document_name": "a.md", "page": 1, "content": "chunk"}],
    )
    llm = _llm_returning("answer")

    with patch("src.docarag.services.agent.get_chat_model", return_value=llm):
        result = await generate_answer_node(state)

    assert result == {"answer": "answer"}
    messages = llm.ainvoke.call_args.args[0]
    assert [type(m) for m in messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert "Сводка." in messages[0].content
    assert "Не пиши вступлений" in messages[0].content
    assert messages[2].content == "x" * 8 + "…"
    assert "[Фрагмент 1, источник a.md, раздел 1]\nchunk" in messages[3].content
    assert (
        messages[3]
        .content.rstrip()
        .endswith("Вопрос оператора: А как его подключить?\n\nОтвет:")
    )


@pytest.mark.asyncio
async def test_evaluate_answer_node_scores_the_standalone_query():
    """Test that the evaluator sees the resolved question, not the bare follow-up."""
    from src.docarag.services.agent import evaluate_answer_node

    state = AgentState(
        query="а для юрлиц?",
        rephrased_query="условия подключения ELAN для юридических лиц",
        answer="...",
        retrieved_docs=[{"content": "c"}],
    )
    llm = _llm_returning("0.9")

    with patch("src.docarag.services.agent.get_chat_model", return_value=llm):
        result = await evaluate_answer_node(state)

    prompt = llm.ainvoke.call_args.args[0]
    assert "условия подключения ELAN для юридических лиц" in prompt
    assert "а для юрлиц?" not in prompt
    assert result["confidence"] == 0.9


@pytest.mark.asyncio
async def test_query_documents_with_session_loads_memory_and_records_turn():
    """Test the wiring around the graph: memory in, both turns out, id echoed."""
    from src.docarag.models.requests import QueryRequest
    from src.docarag.models.sessions import SessionMemory
    from src.docarag.services.agent import query_documents

    memory = SessionMemory(
        session_id="chat-1",
        recent=_history(),
        summary="Сводка.",
        total_messages=2,
        available=True,
    )
    graph = Mock()
    graph.ainvoke = AsyncMock(
        return_value={
            "answer": "ok",
            "rephrased_query": "Подключение ELAN",
            "confidence": 0.8,
            "iterations": 1,
            "retrieved_docs": [
                {"document_name": "a.md", "domain": "elan", "content": "c"}
            ],
        }
    )
    record = AsyncMock(return_value=True)
    schedule = Mock()

    with (
        patch("src.docarag.services.agent.load_memory", AsyncMock(return_value=memory)),
        patch("src.docarag.services.agent.build_agent_graph", return_value=graph),
        patch("src.docarag.services.agent.record_turn", record),
        patch("src.docarag.services.agent.schedule_summary_refresh", schedule),
    ):
        response = await query_documents(
            QueryRequest(query="А как его подключить?", session_id="chat-1")
        )

    initial_state = graph.ainvoke.call_args.args[0]
    assert initial_state.session_id == "chat-1"
    assert initial_state.history == memory.recent
    assert initial_state.history_summary == "Сводка."
    assert response.session_id == "chat-1"
    assert response.rephrased_query == "Подключение ELAN"
    record.assert_awaited_once()
    assert record.call_args.args[:3] == ("chat-1", memory, "А как его подключить?")
    assert record.call_args.args[3] is response
    schedule.assert_called_once_with("chat-1", memory)


@pytest.mark.asyncio
async def test_query_documents_without_session_is_stateless():
    """Test that scripts without a session id never touch the session store."""
    from src.docarag.models.requests import QueryRequest
    from src.docarag.services.agent import query_documents

    graph = Mock()
    graph.ainvoke = AsyncMock(
        return_value={"answer": "ok", "confidence": 0.5, "iterations": 1}
    )
    schedule = Mock()

    with (
        patch("src.docarag.services.agent.build_agent_graph", return_value=graph),
        patch("src.docarag.services.sessions.get_session_store") as get_store,
        patch("src.docarag.services.agent.schedule_summary_refresh", schedule),
    ):
        response = await query_documents(QueryRequest(query="q"))

    get_store.assert_not_called()
    schedule.assert_not_called()
    assert response.session_id is None
    assert graph.ainvoke.call_args.args[0].history == []
