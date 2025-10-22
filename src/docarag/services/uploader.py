from typing import Tuple, Dict, Any, Optional
import uuid
import magic
import httpx
from urllib.parse import urlparse
from src.docarag.settings import settings
from src.docarag.clients.minio_client import (
    get_minio_client,
    ensure_bucket_exists,
    upload_file_to_minio,
)
from src.docarag.consts import SUPPORTED_MIME_TYPES

# Size of initial chunk to download for MIME detection (8KB is enough for magic numbers)
MIME_DETECTION_CHUNK_SIZE = 8192


def detect_file_type_from_header(content_type: str) -> Optional[str]:
    """
    Detect file type from Content-Type header.

    Args:
        content_type: Content-Type header value

    Returns:
        Normalized file type (pdf, doc, docx) or None if not supported/recognized
    """
    # Clean up content type (remove charset and other parameters)
    mime_type = content_type.split(";")[0].strip().lower()
    return SUPPORTED_MIME_TYPES.get(mime_type)


def detect_file_type(file_content: bytes) -> str:
    """
    Detect file type using python-magic from file content.
    Only needs the first few KB of the file for detection.

    Args:
        file_content: File content as bytes (can be partial, minimum 8KB recommended)

    Returns:
        Normalized file type (pdf, doc, docx)

    Raises:
        ValueError: If file type is not supported
    """
    try:
        # Use only first chunk for detection if content is large
        detection_sample = file_content[:MIME_DETECTION_CHUNK_SIZE]
        mime = magic.from_buffer(detection_sample, mime=True)

        if mime not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported file type: {mime}. " f"Supported types: PDF, DOC, DOCX"
            )

        return SUPPORTED_MIME_TYPES[mime]

    except Exception as e:
        raise ValueError(f"Failed to detect file type: {str(e)}")


async def download_file_from_url(url: str) -> Tuple[bytes, str, str]:
    """
    Download file from URL using httpx with optimized MIME detection.
    
    First checks Content-Type header, then downloads only initial chunk
    for MIME detection before downloading the full file.

    Args:
        url: URL to download from

    Returns:
        Tuple of (file_content, filename, detected_type)

    Raises:
        ValueError: If file type is not supported
        Exception: If download fails
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, make a HEAD request to check Content-Type header
            head_response = await client.head(url, follow_redirects=True)
            head_response.raise_for_status()
            
            content_type = head_response.headers.get("content-type", "")
            detected_type_from_header = detect_file_type_from_header(content_type)
            
            # Try to extract filename from Content-Disposition header
            content_disposition = head_response.headers.get("content-disposition", "")
            filename = None
            
            if "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"')
            
            # Fall back to URL path
            if not filename:
                parsed_url = urlparse(url)
                filename = parsed_url.path.split("/")[-1] or "downloaded_file"
            
            # If we got a valid type from header, we can proceed with full download
            # Otherwise, download first chunk to detect MIME type
            if detected_type_from_header:
                # Content-Type header is reliable, download full file
                response = await client.get(url)
                response.raise_for_status()
                return response.content, filename, detected_type_from_header
            else:
                # Need to check actual file content
                # Download first chunk for MIME detection
                headers = {"Range": f"bytes=0-{MIME_DETECTION_CHUNK_SIZE - 1}"}
                chunk_response = await client.get(url, headers=headers)
                
                # Some servers don't support Range requests, fall back to full download
                if chunk_response.status_code == 206:  # Partial Content
                    # Server supports range requests
                    first_chunk = chunk_response.content
                    detected_type = detect_file_type(first_chunk)
                    
                    # Now download the full file
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.content, filename, detected_type
                else:
                    # Server doesn't support range, we already have full content
                    chunk_response.raise_for_status()
                    file_content = chunk_response.content
                    detected_type = detect_file_type(file_content)
                    return file_content, filename, detected_type

    except ValueError:
        # Re-raise validation errors (unsupported file type)
        raise
    except httpx.HTTPError as e:
        raise Exception(f"Failed to download file from URL: {str(e)}")
    except Exception as e:
        raise Exception(f"Error downloading file: {str(e)}")


def upload_document(
    file_content: bytes,
    filename: str,
    file_id: str,
    detected_type: str,
) -> Dict[str, Any]:
    """
    Upload document to MinIO storage.

    Args:
        file_content: File content as bytes
        filename: Original filename
        file_id: Unique file identifier
        detected_type: Detected file type (pdf, doc, docx)

    Returns:
        Dictionary with upload results

    Raises:
        Exception: If upload fails
    """
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise ValueError(
            f"File too large: {file_size_mb:.2f}MB. "
            f"Maximum size is {settings.max_file_size_mb}MB."
        )

    client = get_minio_client()
    ensure_bucket_exists(client, settings.minio_bucket)

    content_type_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
    }
    content_type = content_type_map.get(detected_type, "application/octet-stream")

    object_key = upload_file_to_minio(
        client=client,
        bucket=settings.minio_bucket,
        file_id=file_id,
        file_content=file_content,
        filename=filename,
        content_type=content_type,
        metadata={"type": detected_type},
    )

    return {
        "file_id": file_id,
        "object_key": object_key,
        "filename": filename,
        "file_type": detected_type,
        "size_bytes": len(file_content),
        "content_type": content_type,
    }


async def process_upload(upload_model) -> Dict[str, Any]:
    """
    Process document upload from either direct file or URL.

    This function orchestrates the entire upload flow:
    1. Determines source (file upload vs URL)
    2. Detects file type early (from header or first chunk)
    3. Gets full file content and filename
    4. Generates unique file ID
    5. Uploads to MinIO storage

    Args:
        upload_model: UploadModel instance with validated upload data

    Returns:
        Dictionary with upload results containing:
            - file_id: Unique file identifier
            - object_key: MinIO object key
            - filename: Original filename
            - file_type: Detected file type
            - size_bytes: File size in bytes
            - content_type: MIME content type

    Raises:
        ValueError: If file type is unsupported or validation fails
        Exception: If download or upload fails
    """
    
    if upload_model.document is not None:
        filename = upload_model.document_name
        first_chunk = await upload_model.document.read(MIME_DETECTION_CHUNK_SIZE)
        detected_type = detect_file_type(first_chunk)
        remaining_content = await upload_model.document.read()
        file_content = first_chunk + remaining_content
    else:
        file_content, filename, detected_type = await download_file_from_url(
            str(upload_model.document_url)
        )        
        if upload_model.document_name and upload_model.document_name != filename:
            ext = filename.split(".")[-1] if "." in filename else ""
            filename = (
                f"{upload_model.document_name}.{ext}"
                if ext
                else upload_model.document_name
            )

    file_id = str(uuid.uuid4())

    upload_result = upload_document(
        file_content=file_content,
        filename=filename,
        file_id=file_id,
        detected_type=detected_type,
    )

    return upload_result
