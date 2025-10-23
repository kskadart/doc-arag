"""FastAPI dependency injection functions."""

from typing import Optional, Callable, List, Dict
from fastapi import Form, File, UploadFile, HTTPException
from src.docarag.models.upload import UploadModel
from src.docarag.clients import get_minio_client, list_all_files, download_file_by_id
from src.docarag.services.parsers import parse_document
from src.docarag.settings import settings


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


def get_all_files() -> list[dict]:
    """
    Dependency function to retrieve all files from MinIO storage.

    Returns:
        List of dictionaries containing file information and metadata

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        client = get_minio_client()
        return list_all_files(client, settings.minio_bucket)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving files from storage: {str(e)}",
        )


def file_downloader() -> Callable[[str], tuple[bytes, str, dict]]:
    """
    Dependency function to get a file downloader function.

    Returns:
        A callable that downloads a file by document_id
        Signature: (document_id: str) -> tuple[bytes, str, dict]
        Returns (file_content, filename, metadata) where metadata includes content_type

    Raises:
        HTTPException: If downloader initialization fails
    """

    def download(document_id: str) -> tuple[bytes, str, dict]:
        try:
            client = get_minio_client()
            return download_file_by_id(client, settings.minio_bucket, document_id)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error downloading file: {str(e)}",
            )

    return download


def parse_document_dependency() -> Callable[[bytes, str], List[Dict[str, str | int]]]:
    """
    Dependency function to get a document parser function.

    Returns:
        A callable that parses a document and returns chunks
        Signature: (file_content: bytes, content_type: str) -> List[Dict[str, str | int]]

    Raises:
        HTTPException: If parser initialization fails
    """

    def parser(file_content: bytes, content_type: str) -> List[Dict[str, str | int]]:
        try:
            return parse_document(
                file_content=file_content,
                content_type=content_type,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing document: {str(e)}",
            )

    return parser
