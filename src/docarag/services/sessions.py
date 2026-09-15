"""
Conversational sessions: what the operator asked earlier in the same chat.

The LangGraph agent stays stateless; this module loads memory before a run,
persists both turns after it and keeps a rolling summary of older turns. The
store is a small protocol with a Weaviate implementation (persistent, shared
between workers) and an in-memory one for tests and quick runs.
"""

import asyncio
import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from weaviate.classes.config import Configure
from weaviate.classes.query import Filter, Sort
from weaviate.exceptions import UnexpectedStatusCodeError
from weaviate.util import generate_uuid5

from src.docarag.clients.vector_db_client import (
    get_vector_db_client,
    is_collection_already_exists_error,
)
from src.docarag.consts import SESSION_COLLECTION_NAME, SESSION_SUMMARY_ROLE
from src.docarag.errors import SessionStoreError
from src.docarag.models.responses import AgentQueryResponse
from src.docarag.models.sessions import ChatTurn, SessionMemory, StoredSession
from src.docarag.services.llm import get_chat_model
from src.docarag.settings import settings
from src.docarag.utils.session_collection_conf import (
    SESSION_COLLECTION_DESCRIPTION,
    SESSION_COLLECTION_PROPERTIES,
)

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = "…"
_ROLE_LABELS = {"user": "Operator", "assistant": "Assistant"}
# delete_many is capped by QUERY_MAXIMUM_RESULTS (10000) per call
_SWEEP_MAX_ROUNDS = 20


# --- Store protocol -----------------------------------------------------------


class SessionStore(Protocol):
    """Persistence of chat turns and the per-session rolling summary."""

    async def load(self, session_id: str, limit: int) -> StoredSession | None:
        """Newest `limit` messages (returned oldest first) plus the summary row."""
        ...

    async def append(self, session_id: str, turns: Sequence[ChatTurn]) -> None: ...

    async def save_summary(
        self, session_id: str, summary: str, covers_messages: int
    ) -> None: ...

    async def delete(self, session_id: str) -> int:
        """Remove every row of the session; returns the number deleted."""
        ...

    async def sweep_expired(self, cutoff: datetime) -> int:
        """Remove rows whose `updated_at` is before `cutoff`; returns the number deleted."""
        ...


def summary_uuid(session_id: str) -> str:
    """Deterministic id of the summary row, so it is rewritten in place."""
    return generate_uuid5(f"{session_id}/{SESSION_SUMMARY_ROLE}")


class InMemorySessionStore:
    """Process-local store: tests, dry runs and SESSION_STORE=memory."""

    def __init__(self) -> None:
        self._messages: dict[str, list[ChatTurn]] = {}
        # session_id -> (summary, covers_messages, updated_at)
        self._summaries: dict[str, tuple[str, int, datetime]] = {}
        self._lock = asyncio.Lock()

    async def load(self, session_id: str, limit: int) -> StoredSession | None:
        async with self._lock:
            messages = sorted(
                self._messages.get(session_id, []),
                key=lambda turn: (turn.turn_index, turn.created_at),
            )[-limit:]
            summary = self._summaries.get(session_id)
        if not messages and summary is None:
            return None
        return StoredSession(
            session_id=session_id,
            messages=messages,
            summary=summary[0] if summary else None,
            summary_covers_messages=summary[1] if summary else 0,
            created_at=messages[0].created_at if messages else None,
            updated_at=messages[-1].created_at
            if messages
            else summary[2]
            if summary
            else None,
        )

    async def append(self, session_id: str, turns: Sequence[ChatTurn]) -> None:
        async with self._lock:
            self._messages.setdefault(session_id, []).extend(turns)

    async def save_summary(
        self, session_id: str, summary: str, covers_messages: int
    ) -> None:
        async with self._lock:
            self._summaries[session_id] = (summary, covers_messages, datetime.now(UTC))

    async def delete(self, session_id: str) -> int:
        async with self._lock:
            deleted = len(self._messages.pop(session_id, []))
            if self._summaries.pop(session_id, None) is not None:
                deleted += 1
        return deleted

    async def sweep_expired(self, cutoff: datetime) -> int:
        deleted = 0
        async with self._lock:
            for session_id, turns in list(self._messages.items()):
                kept = [turn for turn in turns if turn.created_at >= cutoff]
                deleted += len(turns) - len(kept)
                if kept:
                    self._messages[session_id] = kept
                else:
                    del self._messages[session_id]
            for session_id, (_, _, updated_at) in list(self._summaries.items()):
                if updated_at < cutoff:
                    del self._summaries[session_id]
                    deleted += 1
        return deleted


class WeaviateSessionStore:
    """Rows in the `ChatMessages` collection; every failure becomes SessionStoreError."""

    async def load(self, session_id: str, limit: int) -> StoredSession | None:
        try:
            async with get_vector_db_client() as client:
                collection = client.collections.get(SESSION_COLLECTION_NAME)
                response = await collection.query.fetch_objects(
                    filters=(
                        Filter.by_property("session_id").equal(session_id)
                        & Filter.by_property("role").not_equal(SESSION_SUMMARY_ROLE)
                    ),
                    # Newest first so the limit trims old turns, not the tail
                    sort=Sort.by_property("turn_index", ascending=False).by_property(
                        "created_at", ascending=False
                    ),
                    limit=limit,
                )
                summary_obj = await collection.query.fetch_object_by_id(
                    summary_uuid(session_id)
                )
        except Exception as exc:
            raise SessionStoreError(
                f"Failed to load session {session_id}: {exc}"
            ) from exc

        messages = [
            _turn_from_properties(obj.properties) for obj in reversed(response.objects)
        ]
        summary_props = summary_obj.properties if summary_obj is not None else None
        if not messages and summary_props is None:
            return None
        summary_updated = summary_props.get("updated_at") if summary_props else None
        return StoredSession(
            session_id=session_id,
            messages=messages,
            summary=str(summary_props["content"]) if summary_props else None,
            summary_covers_messages=int(summary_props.get("turn_index") or 0)
            if summary_props
            else 0,
            created_at=messages[0].created_at if messages else None,
            updated_at=messages[-1].created_at if messages else summary_updated,
        )

    async def append(self, session_id: str, turns: Sequence[ChatTurn]) -> None:
        try:
            async with get_vector_db_client() as client:
                collection = client.collections.get(SESSION_COLLECTION_NAME)
                for turn in turns:
                    await collection.data.insert(
                        properties=_turn_to_properties(session_id, turn)
                    )
        except Exception as exc:
            raise SessionStoreError(
                f"Failed to append to session {session_id}: {exc}"
            ) from exc

    async def save_summary(
        self, session_id: str, summary: str, covers_messages: int
    ) -> None:
        now = datetime.now(UTC)
        uuid = summary_uuid(session_id)
        try:
            async with get_vector_db_client() as client:
                collection = client.collections.get(SESSION_COLLECTION_NAME)
                if await collection.data.exists(uuid):
                    await collection.data.update(
                        uuid=uuid,
                        properties={
                            "content": summary,
                            "turn_index": covers_messages,
                            "updated_at": now,
                        },
                    )
                else:
                    await collection.data.insert(
                        uuid=uuid,
                        properties={
                            "session_id": session_id,
                            "role": SESSION_SUMMARY_ROLE,
                            "content": summary,
                            "turn_index": covers_messages,
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
        except Exception as exc:
            raise SessionStoreError(
                f"Failed to save summary of session {session_id}: {exc}"
            ) from exc

    async def delete(self, session_id: str) -> int:
        try:
            async with get_vector_db_client() as client:
                collection = client.collections.get(SESSION_COLLECTION_NAME)
                result = await collection.data.delete_many(
                    where=Filter.by_property("session_id").equal(session_id)
                )
        except Exception as exc:
            raise SessionStoreError(
                f"Failed to delete session {session_id}: {exc}"
            ) from exc
        return int(result.successful)

    async def sweep_expired(self, cutoff: datetime) -> int:
        deleted = 0
        try:
            async with get_vector_db_client() as client:
                collection = client.collections.get(SESSION_COLLECTION_NAME)
                for _ in range(_SWEEP_MAX_ROUNDS):
                    result = await collection.data.delete_many(
                        where=Filter.by_property("updated_at").less_than(cutoff)
                    )
                    deleted += int(result.successful)
                    if result.matches == 0 or result.successful == 0:
                        break
        except Exception as exc:
            raise SessionStoreError(f"Failed to sweep expired sessions: {exc}") from exc
        return deleted


def _turn_to_properties(session_id: str, turn: ChatTurn) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "role": turn.role,
        "content": turn.content,
        "turn_index": turn.turn_index,
        "created_at": turn.created_at,
        "updated_at": turn.created_at,
        "standalone_query": turn.standalone_query,
        "confidence": turn.confidence,
        "source_documents": list(turn.source_documents),
        "source_domains": list(turn.source_domains),
    }


def _turn_from_properties(props: dict[str, Any]) -> ChatTurn:
    return ChatTurn(
        role=props.get("role", "user"),
        content=str(props.get("content") or ""),
        created_at=props["created_at"],
        turn_index=int(props.get("turn_index") or 0),
        standalone_query=props.get("standalone_query"),
        confidence=props.get("confidence"),
        source_documents=list(props.get("source_documents") or []),
        source_domains=list(props.get("source_domains") or []),
    )


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """The store selected by SESSION_STORE, built once per process."""
    if settings.session_store == "memory":
        return InMemorySessionStore()
    return WeaviateSessionStore()


# --- Prompt helpers (pure) ------------------------------------------------------


def truncate_message(text: str, max_chars: int) -> str:
    """Cap a message for prompt use; stored content is never truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + TRUNCATION_MARKER


def select_window(
    turns: Sequence[ChatTurn], window: int
) -> tuple[list[ChatTurn], list[ChatTurn]]:
    """Split turns into (older, recent) where recent is the last `window` messages."""
    if window <= 0:
        return list(turns), []
    return list(turns[:-window]), list(turns[-window:])


def history_to_messages(turns: Iterable[ChatTurn], max_chars: int) -> list[BaseMessage]:
    """Verbatim history as LangChain messages for the generation prompt."""
    messages: list[BaseMessage] = []
    for turn in turns:
        content = truncate_message(turn.content, max_chars)
        if turn.role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def format_history_transcript(turns: Iterable[ChatTurn], max_chars: int) -> str:
    """Plain-text transcript for the condense and summary prompts."""
    return "\n".join(
        f"{_ROLE_LABELS.get(turn.role, turn.role)}: "
        f"{truncate_message(turn.content, max_chars)}"
        for turn in turns
    )


def needs_summary_refresh(
    total_messages: int, covered_messages: int, threshold: int, window: int
) -> bool:
    """True when messages older than the verbatim window are not in the summary yet."""
    if total_messages <= threshold:
        return False
    return total_messages - window > covered_messages


# --- Facade used by the agent ---------------------------------------------------


async def load_memory(session_id: str | None) -> SessionMemory:
    """
    Memory for one agent run: the verbatim tail plus the rolling summary.

    Without a session id nothing is read. A store failure is logged and yields
    an unavailable memory, so the query degrades to stateless instead of failing.
    """
    if not session_id:
        return SessionMemory()
    try:
        stored = await get_session_store().load(
            session_id, settings.session_max_stored_messages
        )
    except SessionStoreError as exc:
        logger.warning(f"Session store unavailable, answering without history: {exc}")
        return SessionMemory(session_id=session_id, available=False)
    if stored is None:
        return SessionMemory(session_id=session_id, available=True)
    _, recent = select_window(stored.messages, settings.session_history_messages)
    return SessionMemory(
        session_id=session_id,
        recent=recent,
        summary=stored.summary,
        summary_covers_messages=stored.summary_covers_messages,
        total_messages=stored.total_messages,
        available=True,
    )


def _unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return list(seen)


async def record_turn(
    session_id: str | None,
    memory: SessionMemory,
    question: str,
    response: AgentQueryResponse,
) -> bool:
    """Persist the operator question and the agent answer; True when written."""
    if not session_id or not memory.available:
        return False
    now = datetime.now(UTC)
    base = memory.total_messages
    turns = [
        ChatTurn(role="user", content=question, created_at=now, turn_index=base),
        ChatTurn(
            role="assistant",
            content=response.answer,
            created_at=now,
            turn_index=base + 1,
            standalone_query=response.rephrased_query,
            confidence=response.confidence,
            source_documents=_unique(s.document_name for s in response.sources),
            source_domains=_unique(s.domain for s in response.sources),
        ),
    ]
    try:
        await get_session_store().append(session_id, turns)
    except SessionStoreError as exc:
        logger.warning(f"Could not persist turn of session {session_id}: {exc}")
        return False
    return True


_background_tasks: set[asyncio.Task[None]] = set()


def schedule_summary_refresh(
    session_id: str | None, memory: SessionMemory
) -> asyncio.Task[None] | None:
    """
    Fold turns that left the verbatim window into the summary, in the background.

    The summary is only needed by the next request, so the current answer is
    not delayed by it. The task re-reads the store, which already holds the
    turns just recorded.
    """
    if not session_id or not memory.available:
        return None
    if not needs_summary_refresh(
        memory.total_messages + 2,
        memory.summary_covers_messages,
        settings.session_summary_after_messages,
        settings.session_history_messages,
    ):
        return None
    task = asyncio.create_task(refresh_summary(session_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def refresh_summary(session_id: str) -> None:
    """Regenerate the session summary over every message older than the window."""
    try:
        stored = await get_session_store().load(
            session_id, settings.session_max_stored_messages
        )
        if stored is None:
            return
        cutoff_index = stored.total_messages - settings.session_history_messages
        pending = [
            turn
            for turn in stored.messages
            if stored.summary_covers_messages <= turn.turn_index < cutoff_index
        ]
        if not pending:
            return
        summary = await summarize_history(stored.summary, pending)
        await get_session_store().save_summary(session_id, summary, cutoff_index)
        logger.info(f"Session {session_id}: summary now covers {cutoff_index} messages")
    except Exception as exc:  # fire-and-forget: never surfaces to a request
        logger.warning(f"Summary refresh failed for session {session_id}: {exc}")


SUMMARY_SYSTEM_PROMPT = """You maintain a running summary of a chat between a customer-support operator and a retrieval assistant. The summary is the assistant's only memory of the part of the conversation that no longer fits in the verbatim window.

Rules:
- Keep every fact needed to resolve later follow-ups: the subject discussed (service, tariff, procedure, system), concrete values, decisions taken, questions still open.
- Drop greetings, apologies, filler and anything already superseded.
- Merge the new turns into the existing summary; do not repeat yourself.
- No invention: only what appears in the conversation.
- At most 120 words, plain prose, no headings or bullets.
- Write in the SAME LANGUAGE as the conversation.
Output the updated summary and nothing else."""


async def summarize_history(existing: str | None, turns: Sequence[ChatTurn]) -> str:
    """One LLM call that folds `turns` into `existing`."""
    transcript = format_history_transcript(turns, settings.session_message_max_chars)
    prompt = (
        f"Existing summary:\n{existing or '(none)'}\n\n"
        f"New turns to fold in:\n{transcript}\n\n"
        "Updated summary:"
    )
    response = await get_chat_model(0.2).ainvoke(
        [SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return response.text.strip()


# --- Collection lifecycle -------------------------------------------------------


async def create_session_collection() -> None:
    """Create the ChatMessages collection unless it exists; no-op for the memory store."""
    if settings.session_store != "weaviate":
        return
    async with get_vector_db_client() as client:
        if await client.collections.exists(SESSION_COLLECTION_NAME):
            logger.info(f"Collection {SESSION_COLLECTION_NAME} already exists")
            return
        try:
            await client.collections.create(
                name=SESSION_COLLECTION_NAME,
                description=SESSION_COLLECTION_DESCRIPTION,
                properties=SESSION_COLLECTION_PROPERTIES,
                # Same shape as the document collection; vectors are simply never sent
                vector_config=Configure.Vectors.self_provided(),
            )
        except UnexpectedStatusCodeError as exc:
            # Another uvicorn worker won the startup race
            if not is_collection_already_exists_error(exc):
                raise
            logger.info(
                f"Collection {SESSION_COLLECTION_NAME} was created by another worker"
            )
            return
        logger.info(f"Collection {SESSION_COLLECTION_NAME} created successfully")


async def sweep_expired_sessions() -> int:
    """Delete rows older than SESSION_TTL_DAYS; logs and returns 0 on store failure."""
    cutoff = datetime.now(UTC) - timedelta(days=settings.session_ttl_days)
    try:
        deleted = await get_session_store().sweep_expired(cutoff)
    except SessionStoreError as exc:
        logger.warning(f"Session TTL sweep skipped: {exc}")
        return 0
    if deleted:
        logger.info(f"Session TTL sweep removed {deleted} row(s) older than {cutoff}")
    return deleted


async def run_session_cleanup_loop() -> None:
    """Periodic TTL sweep; cancelled by the application lifespan on shutdown."""
    interval = settings.session_cleanup_interval_minutes * 60
    while True:
        await asyncio.sleep(interval)
        await sweep_expired_sessions()
