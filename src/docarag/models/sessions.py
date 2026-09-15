"""Conversation memory shared by the session store and the RAG agent."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ChatRole = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    """One stored message; assistant turns also carry retrieval metadata."""

    role: ChatRole
    content: str
    created_at: datetime
    turn_index: int = Field(
        default=0, ge=0, description="Dense position in the session"
    )
    standalone_query: str | None = Field(
        default=None, description="Assistant only: the query that was actually embedded"
    )
    confidence: float | None = Field(default=None, description="Assistant only")
    source_documents: list[str] = Field(
        default_factory=list, description="Assistant only: documents used as context"
    )
    source_domains: list[str] = Field(
        default_factory=list, description="Assistant only: domains of those documents"
    )


class SessionMemory(BaseModel):
    """What the agent is given: the verbatim tail plus a summary of everything older."""

    session_id: str | None = None
    recent: list[ChatTurn] = Field(default_factory=list)
    summary: str | None = None
    summary_covers_messages: int = Field(
        default=0,
        ge=0,
        description="Messages with turn_index below this are summarised",
    )
    total_messages: int = Field(default=0, ge=0)
    # False when there is no session or the store could not be read; nothing is
    # appended then, so turn indexes never collide with rows we failed to see
    available: bool = False


class StoredSession(BaseModel):
    """Full read of one session, ordered oldest first."""

    session_id: str
    messages: list[ChatTurn] = Field(default_factory=list)
    summary: str | None = None
    summary_covers_messages: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def total_messages(self) -> int:
        """Messages ever written, derived from indexes so TTL gaps do not shift it."""
        newest = self.messages[-1].turn_index + 1 if self.messages else 0
        return max(newest, self.summary_covers_messages)
