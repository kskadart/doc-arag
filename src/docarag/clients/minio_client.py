from typing import Optional, Dict
import datetime
from io import BytesIO
from urllib.parse import urlparse
from minio import Minio
from minio.error import S3Error
from src.docarag.settings import settings


def get_minio_client() -> Minio:
    """
    Get configured MinIO client.

    Returns:
        Configured Minio client

    Raises:
        Exception: If client initialization fails
    """
    try:
        # Parse endpoint to extract host and determine if secure
        parsed_endpoint = urlparse(settings.minio_endpoint)

        # Extract host (with port if present)
        if parsed_endpoint.netloc:
            endpoint = parsed_endpoint.netloc
        else:
            # If no scheme was provided, use the endpoint as-is
            endpoint = settings.minio_endpoint

        # Determine if connection should be secure
        secure = settings.minio_secure
        if parsed_endpoint.scheme:
            secure = parsed_endpoint.scheme == "https"

        client = Minio(
            endpoint=endpoint,
            access_key=settings.minio_access_key.get_secret_value(),
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=secure,
        )
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize MinIO client: {str(e)}")


def ensure_bucket_exists(client: Minio, bucket: str) -> None:
    """
    Ensure MinIO bucket exists, create if it doesn't.

    Args:
        client: Minio client
        bucket: Bucket name

    Raises:
        Exception: If bucket creation fails
    """
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    except S3Error as e:
        raise Exception(f"Failed to create bucket: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to check/create bucket: {str(e)}")


def upload_file_to_minio(
    client: Minio,
    bucket: str,
    file_id: str,
    file_content: bytes,
    filename: str,
    content_type: str,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Upload file to MinIO.

    Args:
        client: Minio client
        bucket: Bucket name
        file_id: Unique file identifier
        file_content: File content as bytes
        filename: Original filename
        content_type: MIME type
        metadata: Optional metadata dictionary

    Returns:
        Object key in MinIO

    Raises:
        Exception: If upload fails
    """
    try:
        object_key = f"{file_id}/{filename}"

        minio_metadata = metadata or {}
        minio_metadata.update(
            {
                "filename": filename,
                "upload_timestamp": datetime.datetime.now(datetime.UTC),
            }
        )

        file_data = BytesIO(file_content)
        file_size = len(file_content)

        client.put_object(
            bucket_name=bucket,
            object_name=object_key,
            data=file_data,
            length=file_size,
            content_type=content_type,
            metadata=minio_metadata,
        )

        return object_key

    except S3Error as e:
        raise Exception(f"Failed to upload file to MinIO: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to upload file to MinIO: {str(e)}")


def list_all_files(client: Minio, bucket: str) -> list[Dict]:
    """
    List all files in MinIO bucket with their metadata.

    Args:
        client: Minio client
        bucket: Bucket name

    Returns:
        List of dictionaries containing file information and metadata

    Raises:
        Exception: If listing fails
    """
    try:
        files = []
        objects = client.list_objects(bucket, recursive=True)

        for obj in objects:
            try:
                stat = client.stat_object(bucket, obj.object_name)

                file_id = (
                    obj.object_name.split("/")[0] if "/" in obj.object_name else None
                )
                filename = (
                    obj.object_name.split("/")[-1]
                    if "/" in obj.object_name
                    else obj.object_name
                )

                file_info = {
                    "file_id": file_id,
                    "object_key": obj.object_name,
                    "filename": filename,
                    "size_bytes": obj.size,
                    "content_type": stat.content_type,
                    "last_modified": obj.last_modified,
                    "metadata": stat.metadata or {},
                }

                files.append(file_info)
            except S3Error:
                continue

        return files

    except S3Error as e:
        raise Exception(f"Failed to list files from MinIO: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to list files from MinIO: {str(e)}")


def delete_file_by_id(client: Minio, bucket: str, file_id: str) -> int:
    """
    Delete all files associated with a file_id from MinIO.

    Args:
        client: Minio client
        bucket: Bucket name
        file_id: File identifier to delete

    Returns:
        Number of files deleted

    Raises:
        Exception: If deletion fails
    """
    try:
        objects = client.list_objects(bucket, prefix=f"{file_id}/", recursive=True)

        deleted_count = 0
        for obj in objects:
            try:
                client.remove_object(bucket, obj.object_name)
                deleted_count += 1
            except S3Error:
                continue

        return deleted_count

    except S3Error as e:
        raise Exception(f"Failed to delete files from MinIO: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to delete files from MinIO: {str(e)}")


def download_file_by_id(
    client: Minio, bucket: str, document_id: str
) -> tuple[bytes, str, dict]:
    """
    Download a document by document_id from MinIO.

    Args:
        client: Minio client
        bucket: Bucket name
        document_id: Document identifier

    Returns:
        Tuple of (file_content: bytes, filename: str, metadata: dict)
        where metadata includes content_type and other file metadata

    Raises:
        Exception: If document not found or download fails
    """
    try:
        objects = list(
            client.list_objects(bucket, prefix=f"{document_id}/", recursive=True)
        )

        if not objects:
            raise Exception(f"No document found with ID: {document_id}")

        obj = objects[0]
        stat = client.stat_object(bucket, obj.object_name)
        response = client.get_object(bucket, obj.object_name)
        file_content = response.read()
        response.close()

        filename = (
            obj.object_name.split("/", 1)[1]
            if "/" in obj.object_name
            else obj.object_name
        )

        metadata = {
            "content_type": stat.content_type,
            "filename": filename,
            "size_bytes": obj.size,
            "last_modified": obj.last_modified,
            "metadata": stat.metadata or {},
        }

        return file_content, filename, metadata

    except S3Error as e:
        raise Exception(f"Failed to download file from MinIO: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to download file from MinIO: {str(e)}")
