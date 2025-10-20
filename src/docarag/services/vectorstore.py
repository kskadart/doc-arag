from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from src.docarag.config import settings


class VectorStoreService:
    """Service for managing vector storage in Weaviate."""
    
    def __init__(self):
        """Initialize Weaviate client."""
        self.client = weaviate.connect_to_local(
            host=settings.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(settings.weaviate_url.split(":")[-1]) if ":" in settings.weaviate_url.split("//")[-1] else 8080,
        )
        self.collection_name = settings.weaviate_collection
        self.collection = None
    
    def create_schema(self, embedding_dimension: int) -> None:
        """
        Create or update Weaviate schema.
        
        Args:
            embedding_dimension: Dimension of embedding vectors
        """
        try:
            if self.client.collections.exists(self.collection_name):
                self.collection = self.client.collections.get(self.collection_name)
                return
            
            self.collection = self.client.collections.create(
                name=self.collection_name,
                vector_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="file_id", data_type=DataType.TEXT),
                    Property(name="source_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ],
            )
        
        except Exception as e:
            raise Exception(f"Failed to create schema: {str(e)}")
    
    def add_vectors(
        self,
        file_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        source_type: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add document chunks with embeddings to Weaviate.
        
        Args:
            file_id: Unique file identifier
            chunks: List of text chunks with metadata
            embeddings: List of embedding vectors
            source_type: Type of source (pdf, docx, html)
            filename: Original filename
            metadata: Optional metadata
            
        Returns:
            Number of chunks added
            
        Raises:
            Exception: If insertion fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if self.collection is None:
            raise Exception("Collection not initialized. Call create_schema first.")
        
        try:
            import json
            
            # Prepare objects for batch insertion
            objects = []
            for chunk, embedding in zip(chunks, embeddings):
                obj = {
                    "content": chunk["content"],
                    "file_id": file_id,
                    "source_type": source_type,
                    "chunk_index": chunk["chunk_index"],
                    "filename": filename,
                    "metadata": json.dumps(metadata or {}),
                }
                objects.append((obj, embedding))
            
            # Batch insert
            with self.collection.batch.dynamic() as batch:
                for obj, vector in objects:
                    batch.add_object(
                        properties=obj,
                        vector=vector,
                    )
            
            return len(chunks)
        
        except Exception as e:
            raise Exception(f"Failed to add vectors: {str(e)}")
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        file_id: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Weaviate.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            file_id: Optional filter by file_id
            source_type: Optional filter by source_type
            
        Returns:
            List of matching documents with metadata
            
        Raises:
            Exception: If search fails
        """
        if self.collection is None:
            raise Exception("Collection not initialized. Call create_schema first.")
        
        try:
            # Build filter
            where_filter = None
            if file_id and source_type:
                where_filter = {
                    "operator": "And",
                    "operands": [
                        {"path": ["file_id"], "operator": "Equal", "valueText": file_id},
                        {"path": ["source_type"], "operator": "Equal", "valueText": source_type},
                    ]
                }
            elif file_id:
                where_filter = {"path": ["file_id"], "operator": "Equal", "valueText": file_id}
            elif source_type:
                where_filter = {"path": ["source_type"], "operator": "Equal", "valueText": source_type}
            
            # Perform search
            if where_filter:
                response = self.collection.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True),
                    filters=where_filter,
                )
            else:
                response = self.collection.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True),
                )
            
            # Format results
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content"),
                    "file_id": obj.properties.get("file_id"),
                    "source_type": obj.properties.get("source_type"),
                    "chunk_index": obj.properties.get("chunk_index"),
                    "filename": obj.properties.get("filename"),
                    "distance": obj.metadata.distance if obj.metadata else None,
                    "score": 1 - (obj.metadata.distance or 0) if obj.metadata else 0,
                })
            
            return results
        
        except Exception as e:
            raise Exception(f"Failed to search vectors: {str(e)}")
    
    def delete_by_file_id(self, file_id: str) -> int:
        """
        Delete all vectors for a specific file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Number of vectors deleted
            
        Raises:
            Exception: If deletion fails
        """
        if self.collection is None:
            raise Exception("Collection not initialized. Call create_schema first.")
        
        try:
            result = self.collection.data.delete_many(
                where={"path": ["file_id"], "operator": "Equal", "valueText": file_id}
            )
            return result.successful
        
        except Exception as e:
            raise Exception(f"Failed to delete vectors: {str(e)}")
    
    def get_documents_metadata(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get metadata for all documents.
        
        Args:
            limit: Maximum number of documents
            
        Returns:
            List of document metadata
        """
        if self.collection is None:
            raise Exception("Collection not initialized. Call create_schema first.")
        
        try:
            response = self.collection.query.fetch_objects(limit=limit)
            
            # Group by file_id
            files_map = {}
            for obj in response.objects:
                file_id = obj.properties.get("file_id")
                if file_id not in files_map:
                    files_map[file_id] = {
                        "file_id": file_id,
                        "filename": obj.properties.get("filename"),
                        "source_type": obj.properties.get("source_type"),
                        "chunks_count": 0,
                    }
                files_map[file_id]["chunks_count"] += 1
            
            return list(files_map.values())
        
        except Exception as e:
            raise Exception(f"Failed to get documents metadata: {str(e)}")
    
    def close(self) -> None:
        """Close Weaviate client connection."""
        if self.client:
            self.client.close()


# Global vector store service instance
vectorstore_service: Optional[VectorStoreService] = None


def get_vectorstore_service() -> VectorStoreService:
    """Get or create vector store service instance."""
    global vectorstore_service
    if vectorstore_service is None:
        vectorstore_service = VectorStoreService()
    return vectorstore_service

