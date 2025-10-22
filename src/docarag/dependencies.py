"""FastAPI dependency injection functions."""

from typing import Optional
from fastapi import Form, File, UploadFile, HTTPException
from src.docarag.models.upload import UploadModel


async def upload_dependencies(
    document_name: str = Form(..., max_length=255, description="Name of the document"),
    document: Optional[UploadFile] = File(None),
    document_url: Optional[str] = Form(None, description="URL to download file from"),
) -> UploadModel:
    """
    Dependency function to validate upload request parameters.

    Args:
        document_name: Name of the document
        document: Optional file upload
        document_url: Optional URL to download file from

    Returns:
        Validated UploadModel instance

    Raises:
        HTTPException: If validation fails
    """
    try:
        # Create and validate the model
        upload_model = UploadModel(
            document_name=document_name,
            document=document,
            document_url=document_url,
        )
        return upload_model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid upload request: {str(e)}",
        )
