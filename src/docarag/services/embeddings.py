"""Embedding façade over the configured OpenAI-compatible endpoint."""

from src.docarag.clients.embedding_http import EmbeddingHTTPClient


class EmbeddingService:
    """Service for generating embeddings through the HTTP embedding client."""

    def __init__(self, client: EmbeddingHTTPClient | None = None) -> None:
        """
        Initialize embedding service.

        Args:
            client: Optional client instance (creates one from settings if omitted)
        """
        self.client = client or EmbeddingHTTPClient()

    async def embed_text_async(self, text: str) -> list[float]:
        """Embed a single text. Raises ValueError on empty text, EmbeddingError on failure."""
        return await self.client.embed_text_async(text)

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts in order. Raises ValueError on empty list, EmbeddingError on failure."""
        return await self.client.embed_batch_async(texts)

    async def get_embedding_dimension_async(self) -> int:
        """Return the vector dimension produced by the configured model."""
        return await self.client.get_embedding_dimension_async()

    async def close_async(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.close_async()


# Global embedding service instance
embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance."""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service
