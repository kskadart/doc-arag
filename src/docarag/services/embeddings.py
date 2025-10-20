from typing import List, Optional
from src.docarag.clients.embedding import EmbeddingGRPCClient


class EmbeddingService:
    """Service for generating embeddings using external gRPC embedding service."""
    
    def __init__(self, client: Optional[EmbeddingGRPCClient] = None):
        """
        Initialize embedding service.
        
        Args:
            client: Optional gRPC client instance (creates new one if not provided)
        """
        self.client = client or EmbeddingGRPCClient()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If text is empty
            Exception: If embedding fails
        """
        return self.client.embed_text(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (ignored for gRPC service)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty
            Exception: If embedding fails
        """
        return self.client.embed_batch(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            Embedding dimension
        """
        return self.client.get_embedding_dimension()
    
    async def embed_text_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using async call.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If text is empty
            Exception: If embedding fails
        """
        return await self.client.embed_text_async(text)
    
    async def embed_batch_async(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using async call.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (ignored for gRPC service)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty
            Exception: If embedding fails
        """
        return await self.client.embed_batch_async(texts)
    
    async def get_embedding_dimension_async(self) -> int:
        """
        Get the dimension of embeddings produced by this service using async call.
        
        Returns:
            Embedding dimension
        """
        return await self.client.get_embedding_dimension_async()
    
    def close(self) -> None:
        """Close the gRPC client connection."""
        self.client.close()
    
    async def close_async(self) -> None:
        """Close the gRPC client connection (async)."""
        await self.client.close_async()


# Global embedding service instance
embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance."""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service

