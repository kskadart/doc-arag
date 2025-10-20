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
    
    file_id: str = Field(..., description="File identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    chunks_processed: Optional[int] = Field(default=None, description="Number of chunks embedded")


class Source(BaseModel):
    """Source document information."""
    
    file_id: str = Field(..., description="File identifier")
    content: str = Field(..., description="Relevant content chunk")
    score: float = Field(..., description="Relevance score")
    source_type: str = Field(..., description="Type of source (pdf, docx, html)")
    chunk_index: int = Field(..., description="Index of the chunk in the document")


class QueryResponse(BaseModel):
    """Response for RAG query."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source documents used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    iterations: int = Field(..., description="Number of agent iterations")
    rephrased_query: Optional[str] = Field(default=None, description="Rephrased version of the query")


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

