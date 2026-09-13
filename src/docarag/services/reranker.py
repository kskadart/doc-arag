"""Reranking service: picks the configured provider and normalises its errors."""

import logging
from typing import Any

import grpc

from src.docarag.clients.reranker_client import RerankerGRPCClient
from src.docarag.clients.reranker_http import RerankerHTTPClient, RerankResult
from src.docarag.errors import RerankerError
from src.docarag.settings import settings

logger = logging.getLogger(__name__)

RerankerClient = RerankerGRPCClient | RerankerHTTPClient


def build_reranker_client() -> RerankerClient | None:
    """Create the client for `settings.reranker_provider`; `None` when disabled."""
    if settings.reranker_provider == "grpc":
        return RerankerGRPCClient()
    if settings.reranker_provider == "openai-rerank":
        return RerankerHTTPClient()
    return None


class RerankerService:
    """Service for reranking documents through the configured provider."""

    def __init__(self, client: RerankerClient | None = None) -> None:
        """
        Initialize reranker service.

        Args:
            client: Optional client instance (built from settings when omitted)
        """
        self.client = client if client is not None else build_reranker_client()

    async def _score(
        self, query: str, texts: list[str], top_n: int
    ) -> list[RerankResult]:
        """Call the provider and convert its answer into `(index, score)` pairs."""
        if self.client is None:
            raise RerankerError("Reranker is disabled (reranker_provider=none)")
        try:
            if isinstance(self.client, RerankerGRPCClient):
                scores = await self.client.rerank_async(query, texts)
                if len(scores) != len(texts):
                    raise RerankerError(
                        f"gRPC reranker returned {len(scores)} scores for {len(texts)} texts"
                    )
                return [
                    RerankResult(index, float(score))
                    for index, score in enumerate(scores)
                ]
            return await self.client.rerank_async(query, texts, top_n)
        except RerankerError:
            raise
        except (grpc.RpcError, ValueError, OSError) as exc:
            raise RerankerError(f"Reranker call failed: {exc}") from exc

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
            RerankerError: If the provider is disabled, unavailable or answers badly
        """
        if not documents:
            return []

        texts = [str(doc.get(content_key, "")) for doc in documents]
        results = await self._score(query, texts, top_k)

        scored_docs = []
        for result in results:
            if not 0 <= result.index < len(documents):
                raise RerankerError(
                    f"Rerank index {result.index} out of range for {len(documents)} documents"
                )
            scored_docs.append(
                {**documents[result.index], "rerank_score": result.relevance_score}
            )
        scored_docs.sort(key=lambda doc: doc["rerank_score"], reverse=True)

        return scored_docs[:top_k]

    async def close_async(self) -> None:
        """Close the underlying client connection."""
        if self.client is not None:
            await self.client.close_async()


# Global reranker service instance
reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    """Get or create reranker service instance."""
    global reranker_service
    if reranker_service is None:
        reranker_service = RerankerService()
    return reranker_service
