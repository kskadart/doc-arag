"""Service for reranking retrieved documents by relevance using the external gRPC reranker."""

import logging
from typing import Any

from src.docarag.clients.reranker_client import RerankerGRPCClient

logger = logging.getLogger(__name__)


class RerankerService:
    """Service for reranking documents using the external gRPC reranker."""

    def __init__(self, client: RerankerGRPCClient | None = None) -> None:
        """
        Initialize reranker service.

        Args:
            client: Optional gRPC client instance (creates new one if not provided)
        """
        self.client = client or RerankerGRPCClient()

    async def rerank_async(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: Search query
            documents: Retrieved documents to rerank, each holding text under `content_key`
            top_k: Number of top-scoring documents to return
            content_key: Key in each document dict holding the text to score

        Returns:
            The `top_k` documents with the highest relevance score, sorted descending,
            each augmented with a `rerank_score` field. Empty input returns an empty list.

        Raises:
            ValueError: If query is empty
            grpc.RpcError: If the reranker service is unavailable or times out
        """
        if not documents:
            return []

        texts = [doc.get(content_key, "") for doc in documents]
        scores = await self.client.rerank_async(query, texts)

        scored_docs = [
            {**doc, "rerank_score": score}
            for doc, score in zip(documents, scores, strict=True)
        ]
        scored_docs.sort(key=lambda doc: doc["rerank_score"], reverse=True)

        return scored_docs[:top_k]

    async def close_async(self) -> None:
        """Close the gRPC client connection."""
        await self.client.close_async()


# Global reranker service instance
reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    """Get or create reranker service instance."""
    global reranker_service
    if reranker_service is None:
        reranker_service = RerankerService()
    return reranker_service
