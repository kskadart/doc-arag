from typing import Optional
from fastapi import UploadFile
from pydantic import BaseModel, Field, model_validator


class UploadModel(BaseModel):
    """Model for document upload with validation."""

    document_name: str = Field(..., max_length=255, description="Name of the document")
    document: Optional[UploadFile] = Field(None, description="File upload")
    document_url: Optional[str] = Field(None, description="URL to download file from")

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
