import grpc
import grpc.aio
from typing import List, Optional, Union
import logging

from src.docarag.settings import settings
from src.docarag.embedding_pb2_grpc import EmbeddingServiceStub
from src.docarag.embedding_pb2 import (
    Empty,
    EmbedTextRequest,
    EmbedBatchRequest,
)

logger = logging.getLogger(__name__)


class EmbeddingGRPCClient:
    """gRPC client for communicating with external embedding service."""

    def __init__(
        self,
        url: Optional[str] = None,
        timeout: Optional[int] = None,
        use_async: Optional[bool] = None,
    ):
        """
        Initialize gRPC embedding client.

        Args:
            url: Embedding service URL (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
            use_async: Whether to use async mode (defaults to config)
        """
        self.url = url or settings.embedding_service_url
        self.timeout = timeout or settings.embedding_service_timeout
        self.use_async = (
            use_async if use_async is not None else settings.embedding_use_async
        )

        self._channel: Optional[Union[grpc.Channel, grpc.aio.Channel]] = None
        self._stub = None
        self._embedding_dimension: Optional[int] = None

    def _get_channel(self) -> Union[grpc.Channel, grpc.aio.Channel]:
        """Get or create gRPC channel."""
        if self._channel is None:
            if self.use_async:
                self._channel = grpc.aio.insecure_channel(self.url)
            else:
                self._channel = grpc.insecure_channel(self.url)
        return self._channel

    def _get_stub(self):
        """Get or create gRPC stub."""
        if self._stub is None:
            channel = self._get_channel()
            self._stub = EmbeddingServiceStub(channel)
        return self._stub

    async def embed_text_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using async gRPC call.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If text is empty
            Exception: If embedding fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            stub = self._get_stub()
            request = EmbedTextRequest(text=text)
            response = await stub.EmbedText(request, timeout=self.timeout)
            return list(response.embedding)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using sync gRPC call.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If text is empty
            Exception: If embedding fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            stub = self._get_stub()
            request = EmbedTextRequest(text=text)
            response = stub.EmbedText(request, timeout=self.timeout)
            return list(response.embedding)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def embed_batch_async(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        pooling_strategy: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using async gRPC call.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for server-side processing (default: 32)
            max_length: Maximum token length per text (default: from settings)
            normalize: Whether to normalize embeddings (default: from settings)
            pooling_strategy: Pooling strategy like "mean", "cls" (default: from settings)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts list is empty
            Exception: If embedding fails
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts to embed")

        # Use settings defaults if not provided
        if max_length is None:
            max_length = settings.embedding_max_length
        if normalize is None:
            normalize = settings.embedding_normalize
        if pooling_strategy is None:
            pooling_strategy = settings.embedding_pooling_strategy

        try:
            stub = self._get_stub()
            request = EmbedBatchRequest(
                texts=valid_texts,
                batch_size=batch_size,
                max_length=max_length,
                normalize=normalize,
                pooling_strategy=pooling_strategy,
            )
            response = await stub.EmbedBatch(request, timeout=self.timeout)
            # EmbeddingVector has a 'vector' field containing the actual floats
            return [list(emb.vector) for emb in response.embeddings]

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        pooling_strategy: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using sync gRPC call.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for server-side processing (default: 32)
            max_length: Maximum token length per text (default: from settings)
            normalize: Whether to normalize embeddings (default: from settings)
            pooling_strategy: Pooling strategy like "mean", "cls" (default: from settings)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts list is empty
            Exception: If embedding fails
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts to embed")

        # Use settings defaults if not provided
        if max_length is None:
            max_length = settings.embedding_max_length
        if normalize is None:
            normalize = settings.embedding_normalize
        if pooling_strategy is None:
            pooling_strategy = settings.embedding_pooling_strategy

        try:
            stub = self._get_stub()
            request = EmbedBatchRequest(
                texts=valid_texts,
                batch_size=batch_size,
                max_length=max_length,
                normalize=normalize,
                pooling_strategy=pooling_strategy,
            )
            response = stub.EmbedBatch(request, timeout=self.timeout)
            # EmbeddingVector has a 'vector' field containing the actual floats
            return [list(emb.vector) for emb in response.embeddings]

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")

    async def get_embedding_dimension_async(self) -> int:
        """
        Get the dimension of embeddings produced by the service using async call.

        Returns:
            Embedding dimension
        """
        if self._embedding_dimension is None:
            try:
                stub = self._get_stub()
                response = await stub.GetEmbeddingDimension(
                    Empty(), timeout=self.timeout
                )
                self._embedding_dimension = response.dimension

            except Exception as e:
                logger.error(f"Failed to get embedding dimension: {str(e)}")
                raise Exception(f"Failed to get embedding dimension: {str(e)}")

        return self._embedding_dimension

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the service using sync call.

        Returns:
            Embedding dimension
        """
        if self._embedding_dimension is None:
            try:
                stub = self._get_stub()
                response = stub.GetEmbeddingDimension(Empty(), timeout=self.timeout)
                self._embedding_dimension = response.dimension

            except Exception as e:
                logger.error(f"Failed to get embedding dimension: {str(e)}")
                raise Exception(f"Failed to get embedding dimension: {str(e)}")

        return self._embedding_dimension

    async def close_async(self) -> None:
        """Close async gRPC channel."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None

    def close(self) -> None:
        """Close sync gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_async()
