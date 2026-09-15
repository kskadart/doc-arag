"""Property layout of the chat-session collection (one object per message)."""

from weaviate.classes.config import DataType, Property


SESSION_COLLECTION_DESCRIPTION = (
    "Chat messages of operator sessions; no vectors, filtered by session_id"
)

SESSION_COLLECTION_PROPERTIES = [
    Property(
        name="session_id",
        data_type=DataType.TEXT,
        description="Client-issued chat identifier",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="role",
        data_type=DataType.TEXT,
        description="user, assistant or summary",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="content",
        data_type=DataType.TEXT,
        description="Message text, or the rolling summary text",
        index_filterable=False,
        index_searchable=False,
    ),
    Property(
        name="turn_index",
        data_type=DataType.INT,
        description="Position in the session; on the summary row, messages covered",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="created_at",
        data_type=DataType.DATE,
        description="When the message was written",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="updated_at",
        data_type=DataType.DATE,
        description="Drives the TTL sweep; equals created_at for messages",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="standalone_query",
        data_type=DataType.TEXT,
        description="Assistant rows: the history-resolved query that was embedded",
        index_filterable=False,
        index_searchable=False,
    ),
    Property(
        name="confidence",
        data_type=DataType.NUMBER,
        description="Assistant rows: evaluator confidence",
        index_filterable=False,
        index_searchable=False,
    ),
    Property(
        name="source_documents",
        data_type=DataType.TEXT_ARRAY,
        description="Assistant rows: documents used as context",
        index_filterable=False,
        index_searchable=False,
    ),
    Property(
        name="source_domains",
        data_type=DataType.TEXT_ARRAY,
        description="Assistant rows: domains of the context documents",
        index_filterable=False,
        index_searchable=False,
    ),
]
