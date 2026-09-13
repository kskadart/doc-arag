from pydantic import BaseModel, HttpUrl, Field, field_validator

from src.docarag.consts import DEFAULT_COLLECTION_NAME


class ScrapeRequest(BaseModel):
    """Request model for web page scraping."""

    url: HttpUrl = Field(..., description="URL of the web page to scrape")
    extract_links: bool = Field(
        default=False, description="Whether to extract links from the page"
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or query to ask")
    domain: str | None = Field(
        default=None,
        description=(
            "Optional knowledge-domain filter for retrieved chunks (e.g. 'diagnostics'). "
            "Empty or the legacy collection name means no filter."
        ),
    )
    max_iterations: int = Field(
        default=2, ge=1, le=5, description="Maximum agent iterations"
    )

    @field_validator("domain", mode="after")
    @classmethod
    def _ignore_legacy_collection_name(cls, value: str | None) -> str | None:
        """Older clients send the collection name in this field; treat it as no filter."""
        if value is None or not value.strip() or value == DEFAULT_COLLECTION_NAME:
            return None
        return value
