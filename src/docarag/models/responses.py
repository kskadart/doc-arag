from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UploadResponse(BaseModel):
    """Response for document upload."""

    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class ScrapeResponse(BaseModel):
    """Response for web scraping."""

    file_id: str = Field(..., description="Unique identifier for the scraped content")
    url: str = Field(..., description="URL that was scraped")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class EmbeddingResponse(BaseModel):
    """Response for embedding generation."""

    task_id: str = Field(..., description="Task identifier for tracking progress")
    file_id: str = Field(..., description="File identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    chunks_processed: Optional[int] = Field(
        default=None, description="Number of chunks embedded"
    )
    parsed_chunks: Optional[List[dict]] = Field(
        default=None, description="Parsed document chunks with content and page numbers"
    )


class Source(BaseModel):
    """Source document information."""

    file_id: str = Field(..., description="File identifier")
    content: str = Field(..., description="Relevant content chunk")
    score: float = Field(..., description="Relevance score")
    source_type: str = Field(..., description="Type of source (pdf, docx, html)")
    chunk_index: int = Field(..., description="Index of the chunk in the document")


class QueryResponse(BaseModel):
    """Response for RAG query."""

    query: str = Field(..., description="Original query text")
    domain: str = Field(..., description="Collection that was searched")
    results: List["VectorSearchResult"] = Field(
        default_factory=list, description="Search results"
    )
    total_results: int = Field(..., description="Total number of results returned")


class AgentQueryResponse(BaseModel):
    """Response for agent-based RAG query with generated answer."""

    query: str = Field(..., description="Original query text")
    answer: str = Field(..., description="Agent-generated answer based on retrieved context")
    rephrased_query: Optional[str] = Field(None, description="Rephrased query used for retrieval")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence score")
    iterations: int = Field(..., ge=0, description="Number of agent iterations performed")
    sources_used: int = Field(..., description="Number of source documents used")


class DocumentResponse(BaseModel):
    """Response for single document information."""

    file_id: str = Field(..., description="File identifier")
    filename: str = Field(..., description="Original filename")
    source_type: str = Field(..., description="Type of document")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(..., description="Upload timestamp")
    chunks_count: int = Field(..., description="Number of chunks")


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    documents: List[DocumentResponse] = Field(default_factory=list)
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class DeleteResponse(BaseModel):
    """Response for document deletion."""

    file_id: str = Field(..., description="Deleted file identifier")
    status: str = Field(..., description="Deletion status")
    message: str = Field(..., description="Status message")


class UploadedFileResponse(BaseModel):
    """Response for single uploaded file information."""

    file_id: str = Field(..., description="File identifier")
    object_key: str = Field(..., description="MinIO object key")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    last_modified: datetime = Field(..., description="Last modified timestamp")
    metadata: dict = Field(default_factory=dict, description="File metadata")


class UploadedFilesListResponse(BaseModel):
    """Response for listing uploaded files."""

    files: List[UploadedFileResponse] = Field(default_factory=list)
    total: int = Field(..., description="Total number of files")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class TaskStatusResponse(BaseModel):
    """Response for task status."""

    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status (processing, completed, failed)")
    file_id: Optional[str] = Field(
        default=None, description="Associated file identifier"
    )
    message: str = Field(..., description="Status message")
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    total_chunks: int = Field(
        default=0, description="Total number of chunks to process"
    )
    created_at: datetime = Field(..., description="Task creation timestamp")
    completed_at: Optional[datetime] = Field(
        default=None, description="Task completion timestamp"
    )


class VectorSearchResult(BaseModel):
    """Single vector search result."""

    uuid: str = Field(..., description="Weaviate object UUID")
    document_name: str = Field(..., description="Name of the document")
    page: int = Field(..., description="Page number within the document")
    content: str = Field(..., description="Text content of the document chunk")
    date_created: datetime = Field(
        ..., description="Date and time the document chunk was created"
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score (distance)"
    )


class VectorSearchResponse(BaseModel):
    """Response for vector similarity search."""

    query: str = Field(..., description="Original query text")
    collection_name: str = Field(..., description="Collection that was searched")
    results: List[VectorSearchResult] = Field(
        default_factory=list, description="Search results"
    )
    total_results: int = Field(..., description="Total number of results returned")
