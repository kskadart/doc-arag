"""gRPC client for the external reranker service."""

import logging

import grpc
import grpc.aio

from src.docarag.reranker_pb2 import Empty, HealthResponse, RerankRequest
from src.docarag.reranker_pb2_grpc import RerankerServiceStub
from src.docarag.settings import settings

logger = logging.getLogger(__name__)


class RerankerGRPCClient:
    """gRPC client for communicating with the external reranker service."""

    def __init__(self, url: str | None = None, timeout: int | None = None) -> None:
        """
        Initialize the async gRPC reranker client.

        Args:
            url: Reranker service URL (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
        """
        self.url = url or settings.reranker_service_url
        self.timeout = timeout or settings.reranker_timeout

        self._channel: grpc.aio.Channel | None = None
        self._stub: RerankerServiceStub | None = None

    def _get_channel(self) -> grpc.aio.Channel:
        """Get or create the async gRPC channel."""
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self.url)
        return self._channel

    def _get_stub(self) -> RerankerServiceStub:
        """Get or create the gRPC stub."""
        if self._stub is None:
            self._stub = RerankerServiceStub(self._get_channel())
        return self._stub

    async def rerank_async(self, query: str, texts: list[str]) -> list[float]:
        """
        Score a query against a batch of candidate texts.

        Args:
            query: Search query
            texts: Candidate texts to score

        Returns:
            Relevance scores, in the same order as `texts`

        Raises:
            ValueError: If query or texts are empty
            grpc.RpcError: If the reranker service is unavailable or times out
        """
        if not query or not query.strip():
            raise ValueError("Cannot rerank with an empty query")
        if not texts:
            raise ValueError("Cannot rerank an empty list of texts")

        stub = self._get_stub()
        request = RerankRequest(query=query, texts=texts)
        response = await stub.Rerank(request, timeout=self.timeout)
        return list(response.scores)

    async def health_check_async(self) -> HealthResponse:
        """
        Check reranker service health.

        Returns:
            Health status, model name and device reported by the service

        Raises:
            grpc.RpcError: If the reranker service is unavailable or times out
        """
        stub = self._get_stub()
        return await stub.HealthCheck(Empty(), timeout=self.timeout)

    async def close_async(self) -> None:
        """Close the async gRPC channel."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def __aenter__(self) -> "RerankerGRPCClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.close_async()
