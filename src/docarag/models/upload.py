from fastapi import UploadFile
from pydantic import BaseModel, Field, model_validator

from src.docarag.consts import DEFAULT_DOMAIN, DOMAIN_MAX_LENGTH, DOMAIN_PATTERN


class UploadModel(BaseModel):
    """Model for document upload with validation."""

    document_name: str = Field(..., max_length=255, description="Name of the document")
    document: UploadFile | None = Field(default=None, description="File upload")
    document_url: str | None = Field(
        default=None, description="URL to download file from"
    )
    domain: str = Field(
        default=DEFAULT_DOMAIN,
        max_length=DOMAIN_MAX_LENGTH,
        pattern=DOMAIN_PATTERN,
        description="Knowledge domain slug the document belongs to",
    )

    @model_validator(mode="after")
    def validate_document_source(self):
        """Ensure exactly one of document or document_url is provided."""
        has_document = self.document is not None
        has_url = self.document_url is not None and len(self.document_url) > 0

        if not has_document and not has_url:
            raise ValueError("Either 'document' or 'document_url' must be provided")

        if has_document and has_url:
            raise ValueError(
                "Only one of 'document' or 'document_url' should be provided, not both"
            )

        return self
