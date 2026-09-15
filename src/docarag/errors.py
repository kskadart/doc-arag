"""Application-level exceptions raised by the model provider clients."""


class DocARAGError(Exception):
    """Base class for doc-arag errors."""


class EmbeddingError(DocARAGError):
    """The embedding endpoint failed or returned an unusable payload."""


class RerankerError(DocARAGError):
    """The reranker failed, timed out or returned an unusable payload."""


class SessionStoreError(DocARAGError):
    """The conversation store failed to read or write a session."""
