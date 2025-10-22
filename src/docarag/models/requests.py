from typing import Optional
from pydantic import BaseModel, HttpUrl, Field


class ScrapeRequest(BaseModel):
    """Request model for web page scraping."""

    url: HttpUrl = Field(..., description="URL of the web page to scrape")
    extract_links: bool = Field(
        default=False, description="Whether to extract links from the page"
    )


class QueryRequest(BaseModel):
    """Request model for RAG query."""

    query: str = Field(..., min_length=1, description="The question or query to ask")
    file_id: Optional[str] = Field(
        default=None, description="Filter results to specific file"
    )
    source_type: Optional[str] = Field(
        default=None, description="Filter by source type (pdf, docx, html)"
    )
    max_iterations: int = Field(
        default=2, ge=1, le=5, description="Maximum agent iterations"
    )
