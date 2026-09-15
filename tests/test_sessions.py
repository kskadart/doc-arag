"""Tests for conversational sessions: stores, prompt helpers and the agent facade."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from weaviate.collections.classes.batch import DeleteManyReturn

from src.docarag.consts import SESSION_SUMMARY_ROLE
from src.docarag.errors import SessionStoreError
from src.docarag.models.responses import AgentQueryResponse, SourceChunk
from src.docarag.models.sessions import ChatTurn, SessionMemory
from src.docarag.services import sessions
from src.docarag.services.sessions import (
    InMemorySessionStore,
    WeaviateSessionStore,
    format_history_transcript,
    history_to_messages,
    load_memory,
    needs_summary_refresh,
    record_turn,
    refresh_summary,
    schedule_summary_refresh,
    select_window,
    summary_uuid,
    truncate_message,
)
from src.docarag.settings import settings

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def _turn(index: int, role: str = "user", content: str | None = None) -> ChatTurn:
    return ChatTurn(
        role=role,  # type: ignore[arg-type]
        content=content or f"message {index}",
        created_at=NOW + timedelta(seconds=index),
        turn_index=index,
    )


def _exchange(count: int) -> list[ChatTurn]:
    return [_turn(i, "user" if i % 2 == 0 else "assistant") for i in range(count)]


# --- pure helpers ---------------------------------------------------------------


def test_truncate_message_caps_and_marks():
    assert truncate_message("short", 10) == "short"
    cut = truncate_message("a" * 20, 10)
    assert cut.startswith("a" * 10) and cut.endswith("…") and len(cut) == 11


def test_select_window_boundaries():
    turns = _exchange(5)
    assert select_window(turns, 2) == (turns[:3], turns[3:])
    assert select_window(turns, 5) == ([], turns)
    assert select_window(turns, 10) == ([], turns)
    assert select_window(turns, 0) == (turns, [])


def test_history_to_messages_maps_roles_in_order():
    turns = [_turn(0, "user", "q"), _turn(1, "assistant", "a" * 50)]
    messages = history_to_messages(turns, max_chars=10)
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert messages[0].content == "q"
    assert messages[1].content == "a" * 10 + "…"


def test_format_history_transcript_labels_roles():
    text = format_history_transcript(
        [_turn(0, "user", "hi"), _turn(1, "assistant", "hello")], max_chars=100
    )
    assert text == "Operator: hi\nAssistant: hello"


@pytest.mark.parametrize(
    "total, covered, expected",
    [
        (12, 0, False),  # at threshold: not yet
        (13, 0, True),  # first message past the threshold
        (13, 7, False),  # everything older than the window already covered
        (15, 7, True),  # two more messages left the window
    ],
)
def test_needs_summary_refresh(total, covered, expected):
    assert needs_summary_refresh(total, covered, threshold=12, window=6) is expected


# --- in-memory store ------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_store_roundtrip_keeps_order_and_limits_to_tail():
    store = InMemorySessionStore()
    await store.append("s1", _exchange(4))
    await store.append("s2", [_turn(0)])

    loaded = await store.load("s1", limit=3)

    assert loaded is not None
    assert [t.turn_index for t in loaded.messages] == [1, 2, 3]
    assert loaded.total_messages == 4
    assert loaded.summary is None
    assert await store.load("missing", limit=10) is None


@pytest.mark.asyncio
async def test_memory_store_summary_and_delete_are_per_session():
    store = InMemorySessionStore()
    await store.append("s1", _exchange(2))
    await store.append("s2", _exchange(2))
    await store.save_summary("s1", "first", 2)
    await store.save_summary("s1", "second", 4)

    loaded = await store.load("s1", limit=10)
    assert loaded is not None
    assert (loaded.summary, loaded.summary_covers_messages) == ("second", 4)

    assert await store.delete("s1") == 3
    assert await store.load("s1", limit=10) is None
    assert await store.load("s2", limit=10) is not None
    assert await store.delete("s1") == 0


@pytest.mark.asyncio
async def test_memory_store_sweep_drops_only_old_rows():
    store = InMemorySessionStore()
    await store.append("old", _exchange(2))
    await store.append("fresh", [_turn(0)])
    cutoff = NOW + timedelta(seconds=1)  # keeps turn_index >= 1 only

    deleted = await store.sweep_expired(cutoff)

    assert deleted == 2
    assert await store.load("fresh", limit=10) is None
    old = await store.load("old", limit=10)
    assert old is not None and [t.turn_index for t in old.messages] == [1]


# --- Weaviate store (mocked client) --------------------------------------------


def _weaviate_client(collection: Mock) -> Mock:
    client = Mock()
    client.collections.get = Mock(return_value=collection)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _row(index: int, role: str = "user", **extra):
    obj = Mock()
    obj.properties = {
        "session_id": "s1",
        "role": role,
        "content": f"message {index}",
        "turn_index": index,
        "created_at": NOW + timedelta(seconds=index),
        "updated_at": NOW + timedelta(seconds=index),
        **extra,
    }
    return obj


@pytest.mark.asyncio
async def test_weaviate_load_reads_newest_first_and_returns_ascending():
    response = Mock()
    # Sorted desc by the server; the store must flip it back
    response.objects = [_row(9, "assistant"), _row(8), _row(7, "assistant")]
    summary = Mock()
    summary.properties = {
        "role": SESSION_SUMMARY_ROLE,
        "content": "so far",
        "turn_index": 4,
        "updated_at": NOW,
    }
    collection = Mock()
    collection.query.fetch_objects = AsyncMock(return_value=response)
    collection.query.fetch_object_by_id = AsyncMock(return_value=summary)

    with patch(
        "src.docarag.services.sessions.get_vector_db_client",
        return_value=_weaviate_client(collection),
    ):
        loaded = await WeaviateSessionStore().load("s1", limit=3)

    kwargs = collection.query.fetch_objects.call_args.kwargs
    assert kwargs["limit"] == 3
    assert kwargs["sort"].sorts[0].prop == "turn_index"
    assert kwargs["sort"].sorts[0].ascending is False
    targets = {f.target: f.value for f in kwargs["filters"].filters}
    assert targets == {"session_id": "s1", "role": SESSION_SUMMARY_ROLE}
    collection.query.fetch_object_by_id.assert_awaited_once_with(summary_uuid("s1"))

    assert loaded is not None
    assert [t.turn_index for t in loaded.messages] == [7, 8, 9]
    assert loaded.total_messages == 10  # gaps after a sweep do not shift it
    assert (loaded.summary, loaded.summary_covers_messages) == ("so far", 4)


@pytest.mark.asyncio
async def test_weaviate_load_returns_none_for_unknown_session():
    response = Mock()
    response.objects = []
    collection = Mock()
    collection.query.fetch_objects = AsyncMock(return_value=response)
    collection.query.fetch_object_by_id = AsyncMock(return_value=None)

    with patch(
        "src.docarag.services.sessions.get_vector_db_client",
        return_value=_weaviate_client(collection),
    ):
        assert await WeaviateSessionStore().load("nope", limit=3) is None


@pytest.mark.asyncio
async def test_weaviate_append_writes_one_row_per_turn_with_tz_aware_dates():
    collection = Mock()
    collection.data.insert = AsyncMock()
    turns = [
        _turn(0, "user", "q"),
        ChatTurn(
            role="assistant",
            content="a",
            created_at=NOW,
            turn_index=1,
            standalone_query="q resolved",
            confidence=0.8,
            source_documents=["a.md"],
            source_domains=["diagnostics"],
        ),
    ]

    with patch(
        "src.docarag.services.sessions.get_vector_db_client",
        return_value=_weaviate_client(collection),
    ):
        await WeaviateSessionStore().append("s1", turns)

    rows = [c.kwargs["properties"] for c in collection.data.insert.call_args_list]
    assert [r["turn_index"] for r in rows] == [0, 1]
    assert all(r["session_id"] == "s1" for r in rows)
    assert rows[0]["created_at"].tzinfo is not None
    assert rows[0]["updated_at"] == rows[0]["created_at"]
    assert rows[1]["source_documents"] == ["a.md"]
    assert rows[1]["standalone_query"] == "q resolved"


@pytest.mark.asyncio
async def test_weaviate_save_summary_updates_in_place_or_inserts():
    collection = Mock()
    collection.data.exists = AsyncMock(return_value=True)
    collection.data.update = AsyncMock()
    collection.data.insert = AsyncMock()
    client = _weaviate_client(collection)

    with patch(
        "src.docarag.services.sessions.get_vector_db_client", return_value=client
    ):
        await WeaviateSessionStore().save_summary("s1", "sum", 6)
        collection.data.exists.return_value = False
        await WeaviateSessionStore().save_summary("s1", "sum", 6)

    update = collection.data.update.call_args.kwargs
    assert update["uuid"] == summary_uuid("s1")
    assert update["properties"]["turn_index"] == 6
    insert = collection.data.insert.call_args.kwargs
    assert insert["uuid"] == summary_uuid("s1")
    assert insert["properties"]["role"] == SESSION_SUMMARY_ROLE


@pytest.mark.asyncio
async def test_weaviate_delete_filters_on_session_id():
    collection = Mock()
    collection.data.delete_many = AsyncMock(
        return_value=DeleteManyReturn(failed=0, matches=5, objects=None, successful=5)
    )

    with patch(
        "src.docarag.services.sessions.get_vector_db_client",
        return_value=_weaviate_client(collection),
    ):
        assert await WeaviateSessionStore().delete("s1") == 5

    where = collection.data.delete_many.call_args.kwargs["where"]
    assert (where.target, where.value) == ("session_id", "s1")


@pytest.mark.asyncio
async def test_weaviate_sweep_loops_until_nothing_matches():
    collection = Mock()
    collection.data.delete_many = AsyncMock(
        side_effect=[
            DeleteManyReturn(failed=0, matches=10000, objects=None, successful=10000),
            DeleteManyReturn(failed=0, matches=3, objects=None, successful=3),
            DeleteManyReturn(failed=0, matches=0, objects=None, successful=0),
        ]
    )
    cutoff = NOW - timedelta(days=7)

    with patch(
        "src.docarag.services.sessions.get_vector_db_client",
        return_value=_weaviate_client(collection),
    ):
        assert await WeaviateSessionStore().sweep_expired(cutoff) == 10003

    assert collection.data.delete_many.await_count == 3
    where = collection.data.delete_many.call_args.kwargs["where"]
    assert where.target == "updated_at"
    assert where.value == cutoff and where.value.tzinfo is not None


@pytest.mark.asyncio
async def test_weaviate_failures_become_session_store_error():
    collection = Mock()
    collection.query.fetch_objects = AsyncMock(side_effect=RuntimeError("down"))

    with (
        patch(
            "src.docarag.services.sessions.get_vector_db_client",
            return_value=_weaviate_client(collection),
        ),
        pytest.raises(SessionStoreError, match="down"),
    ):
        await WeaviateSessionStore().load("s1", limit=3)


# --- facade ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_memory_without_session_never_touches_store():
    with patch("src.docarag.services.sessions.get_session_store") as get_store:
        memory = await load_memory(None)
    get_store.assert_not_called()
    assert memory == SessionMemory()


@pytest.mark.asyncio
async def test_load_memory_windows_history_and_keeps_summary(monkeypatch):
    monkeypatch.setattr(settings, "session_history_messages", 2)
    store = InMemorySessionStore()
    await store.append("s1", _exchange(5))
    await store.save_summary("s1", "earlier", 3)

    with patch("src.docarag.services.sessions.get_session_store", return_value=store):
        memory = await load_memory("s1")

    assert memory.available is True
    assert [t.turn_index for t in memory.recent] == [3, 4]
    assert memory.summary == "earlier"
    assert memory.summary_covers_messages == 3
    assert memory.total_messages == 5


@pytest.mark.asyncio
async def test_load_memory_store_failure_degrades_to_stateless(caplog):
    store = AsyncMock()
    store.load.side_effect = SessionStoreError("weaviate down")

    with (
        patch("src.docarag.services.sessions.get_session_store", return_value=store),
        caplog.at_level(logging.WARNING, logger="src.docarag.services.sessions"),
    ):
        memory = await load_memory("s1")

    assert memory.available is False and memory.recent == []
    assert any("without history" in r.message for r in caplog.records)


def _response(**overrides: Any) -> AgentQueryResponse:
    base: dict[str, Any] = dict(
        query="q",
        answer="the answer",
        rephrased_query="q resolved",
        confidence=0.9,
        iterations=1,
        sources_used=2,
        sources=[
            SourceChunk(
                document_name="a.md",
                domain="diagnostics",
                page=1,
                score=0.9,
                snippet="",
            ),
            SourceChunk(
                document_name="a.md",
                domain="diagnostics",
                page=2,
                score=0.8,
                snippet="",
            ),
            SourceChunk(
                document_name="b.md", domain="billing", page=1, score=0.7, snippet=""
            ),
        ],
    )
    base.update(overrides)
    return AgentQueryResponse(**base)


@pytest.mark.asyncio
async def test_record_turn_appends_both_turns_with_metadata():
    store = InMemorySessionStore()
    memory = SessionMemory(session_id="s1", total_messages=4, available=True)

    with patch("src.docarag.services.sessions.get_session_store", return_value=store):
        assert await record_turn("s1", memory, "why?", _response()) is True

    stored = await store.load("s1", limit=10)
    assert stored is not None
    user, assistant = stored.messages
    assert (user.role, user.content, user.turn_index) == ("user", "why?", 4)
    assert (assistant.role, assistant.turn_index) == ("assistant", 5)
    assert assistant.standalone_query == "q resolved"
    assert assistant.confidence == 0.9
    assert assistant.source_documents == ["a.md", "b.md"]
    assert assistant.source_domains == ["diagnostics", "billing"]


@pytest.mark.asyncio
async def test_record_turn_skips_when_memory_unavailable_or_no_session():
    store = AsyncMock()
    with patch("src.docarag.services.sessions.get_session_store", return_value=store):
        assert (
            await record_turn("s1", SessionMemory(available=False), "q", _response())
            is False
        )
        assert (
            await record_turn(None, SessionMemory(available=True), "q", _response())
            is False
        )
    store.append.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_turn_swallows_store_errors(caplog):
    store = AsyncMock()
    store.append.side_effect = SessionStoreError("write failed")
    memory = SessionMemory(session_id="s1", available=True)

    with (
        patch("src.docarag.services.sessions.get_session_store", return_value=store),
        caplog.at_level(logging.WARNING, logger="src.docarag.services.sessions"),
    ):
        assert await record_turn("s1", memory, "q", _response()) is False

    assert any("Could not persist" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_schedule_summary_refresh_only_past_threshold(monkeypatch):
    monkeypatch.setattr(settings, "session_summary_after_messages", 12)
    monkeypatch.setattr(settings, "session_history_messages", 6)
    refresh = AsyncMock()

    with patch("src.docarag.services.sessions.refresh_summary", refresh):
        below = schedule_summary_refresh(
            "s1", SessionMemory(session_id="s1", total_messages=10, available=True)
        )
        assert below is None
        task = schedule_summary_refresh(
            "s1", SessionMemory(session_id="s1", total_messages=11, available=True)
        )
        assert task is not None
        await task

    refresh.assert_awaited_once_with("s1")
    assert (
        schedule_summary_refresh(
            "s1", SessionMemory(session_id="s1", total_messages=50, available=False)
        )
        is None
    )


@pytest.mark.asyncio
async def test_refresh_summary_folds_uncovered_older_turns(monkeypatch):
    monkeypatch.setattr(settings, "session_history_messages", 2)
    store = InMemorySessionStore()
    await store.append("s1", _exchange(8))
    await store.save_summary("s1", "old", 3)
    summarize = AsyncMock(return_value="new summary")

    with (
        patch("src.docarag.services.sessions.get_session_store", return_value=store),
        patch("src.docarag.services.sessions.summarize_history", summarize),
    ):
        await refresh_summary("s1")

    existing, pending = summarize.call_args.args
    assert existing == "old"
    assert [t.turn_index for t in pending] == [3, 4, 5]  # covered 3, window keeps 6-7
    stored = await store.load("s1", limit=10)
    assert stored is not None
    assert (stored.summary, stored.summary_covers_messages) == ("new summary", 6)


@pytest.mark.asyncio
async def test_refresh_summary_never_raises(caplog):
    store = AsyncMock()
    store.load.side_effect = SessionStoreError("down")

    with (
        patch("src.docarag.services.sessions.get_session_store", return_value=store),
        caplog.at_level(logging.WARNING, logger="src.docarag.services.sessions"),
    ):
        await refresh_summary("s1")

    assert any("Summary refresh failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_sweep_expired_sessions_uses_ttl_and_survives_failures(
    monkeypatch, caplog
):
    monkeypatch.setattr(settings, "session_ttl_days", 7)
    store = AsyncMock()
    store.sweep_expired.return_value = 3

    with patch("src.docarag.services.sessions.get_session_store", return_value=store):
        assert await sessions.sweep_expired_sessions() == 3
        cutoff = store.sweep_expired.call_args.args[0]
        assert cutoff.tzinfo is not None
        assert datetime.now(UTC) - cutoff > timedelta(days=6, hours=23)

        store.sweep_expired.side_effect = SessionStoreError("down")
        with caplog.at_level(logging.WARNING, logger="src.docarag.services.sessions"):
            assert await sessions.sweep_expired_sessions() == 0
    assert any("sweep skipped" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_create_session_collection_is_noop_for_memory_store(monkeypatch):
    monkeypatch.setattr(settings, "session_store", "memory")
    with patch("src.docarag.services.sessions.get_vector_db_client") as get_client:
        await sessions.create_session_collection()
    get_client.assert_not_called()


@pytest.mark.asyncio
async def test_create_session_collection_creates_vectorless_collection(monkeypatch):
    monkeypatch.setattr(settings, "session_store", "weaviate")
    client = Mock()
    client.collections.exists = AsyncMock(return_value=False)
    client.collections.create = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "src.docarag.services.sessions.get_vector_db_client", return_value=client
    ):
        await sessions.create_session_collection()

    kwargs = client.collections.create.call_args.kwargs
    assert kwargs["name"] == "ChatMessages"
    assert {p.name for p in kwargs["properties"]} >= {
        "session_id",
        "role",
        "turn_index",
    }


def test_get_session_store_follows_settings(monkeypatch):
    sessions.get_session_store.cache_clear()
    monkeypatch.setattr(settings, "session_store", "memory")
    assert isinstance(sessions.get_session_store(), InMemorySessionStore)
    sessions.get_session_store.cache_clear()
    monkeypatch.setattr(settings, "session_store", "weaviate")
    assert isinstance(sessions.get_session_store(), WeaviateSessionStore)
    sessions.get_session_store.cache_clear()


def test_summary_uuid_is_deterministic():
    assert summary_uuid("s1") == summary_uuid("s1")
    assert summary_uuid("s1") != summary_uuid("s2")
