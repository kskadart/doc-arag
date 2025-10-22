# from typing import Optional, List, Dict, Any
# import json
# from datetime import datetime, timedelta
# import boto3
# from botocore.exceptions import ClientError
# from src.docarag.settings import settings


# class StorageService:
#     """Service for managing file storage in MinIO/S3."""

#     def __init__(self):
#         """Initialize S3 client for MinIO."""
#         self.client = boto3.client(
#             "s3",
#             endpoint_url=(
#                 f"http://{settings.minio_endpoint}"
#                 if not settings.minio_secure
#                 else f"https://{settings.minio_endpoint}"
#             ),
#             aws_access_key_id=settings.minio_access_key,
#             aws_secret_access_key=settings.minio_secret_key,
#             region_name="us-east-1",
#         )
#         self.bucket = settings.minio_bucket
#         self._ensure_bucket_exists()

#     def _ensure_bucket_exists(self) -> None:
#         """Create bucket if it doesn't exist."""
#         try:
#             self.client.head_bucket(Bucket=self.bucket)
#         except ClientError:
#             try:
#                 self.client.create_bucket(Bucket=self.bucket)
#             except ClientError as e:
#                 raise Exception(f"Failed to create bucket: {str(e)}")

#     def upload_file(
#         self,
#         file_id: str,
#         file_content: bytes,
#         filename: str,
#         content_type: str,
#         metadata: Optional[Dict[str, str]] = None,
#     ) -> str:
#         """
#         Upload file to S3.

#         Args:
#             file_id: Unique file identifier
#             file_content: File content as bytes
#             filename: Original filename
#             content_type: MIME type
#             metadata: Optional metadata dictionary

#         Returns:
#             S3 object key

#         Raises:
#             Exception: If upload fails
#         """
#         try:
#             object_key = f"{file_id}/{filename}"

#             s3_metadata = metadata or {}
#             s3_metadata.update(
#                 {
#                     "filename": filename,
#                     "upload_timestamp": datetime.utcnow().isoformat(),
#                 }
#             )

#             self.client.put_object(
#                 Bucket=self.bucket,
#                 Key=object_key,
#                 Body=file_content,
#                 ContentType=content_type,
#                 Metadata=s3_metadata,
#             )

#             return object_key

#         except ClientError as e:
#             raise Exception(f"Failed to upload file: {str(e)}")

#     def get_file(self, object_key: str) -> bytes:
#         """
#         Retrieve file from S3.

#         Args:
#             object_key: S3 object key

#         Returns:
#             File content as bytes

#         Raises:
#             Exception: If retrieval fails
#         """
#         try:
#             response = self.client.get_object(Bucket=self.bucket, Key=object_key)
#             return response["Body"].read()

#         except ClientError as e:
#             raise Exception(f"Failed to retrieve file: {str(e)}")

#     def get_file_metadata(self, object_key: str) -> Dict[str, Any]:
#         """
#         Get file metadata from S3.

#         Args:
#             object_key: S3 object key

#         Returns:
#             Metadata dictionary

#         Raises:
#             Exception: If retrieval fails
#         """
#         try:
#             response = self.client.head_object(Bucket=self.bucket, Key=object_key)
#             return {
#                 "size": response["ContentLength"],
#                 "content_type": response.get("ContentType"),
#                 "last_modified": response["LastModified"],
#                 "metadata": response.get("Metadata", {}),
#             }

#         except ClientError as e:
#             raise Exception(f"Failed to retrieve metadata: {str(e)}")

#     def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
#         """
#         List files in S3 bucket.

#         Args:
#             prefix: Optional prefix to filter objects

#         Returns:
#             List of file information dictionaries
#         """
#         try:
#             response = self.client.list_objects_v2(
#                 Bucket=self.bucket,
#                 Prefix=prefix,
#             )

#             files = []
#             for obj in response.get("Contents", []):
#                 files.append(
#                     {
#                         "key": obj["Key"],
#                         "size": obj["Size"],
#                         "last_modified": obj["LastModified"],
#                     }
#                 )

#             return files

#         except ClientError as e:
#             raise Exception(f"Failed to list files: {str(e)}")

#     def delete_file(self, object_key: str) -> None:
#         """
#         Delete file from S3.

#         Args:
#             object_key: S3 object key

#         Raises:
#             Exception: If deletion fails
#         """
#         try:
#             self.client.delete_object(Bucket=self.bucket, Key=object_key)

#         except ClientError as e:
#             raise Exception(f"Failed to delete file: {str(e)}")

#     def delete_by_prefix(self, prefix: str) -> int:
#         """
#         Delete all files with given prefix.

#         Args:
#             prefix: Prefix to match (typically file_id)

#         Returns:
#             Number of files deleted

#         Raises:
#             Exception: If deletion fails
#         """
#         try:
#             files = self.list_files(prefix=prefix)

#             if not files:
#                 return 0

#             objects_to_delete = [{"Key": f["key"]} for f in files]

#             self.client.delete_objects(
#                 Bucket=self.bucket,
#                 Delete={"Objects": objects_to_delete},
#             )

#             return len(objects_to_delete)

#         except ClientError as e:
#             raise Exception(f"Failed to delete files: {str(e)}")

#     def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> str:
#         """
#         Generate presigned URL for file access.

#         Args:
#             object_key: S3 object key
#             expiration: URL expiration time in seconds

#         Returns:
#             Presigned URL

#         Raises:
#             Exception: If generation fails
#         """
#         try:
#             url = self.client.generate_presigned_url(
#                 "get_object",
#                 Params={"Bucket": self.bucket, "Key": object_key},
#                 ExpiresIn=expiration,
#             )
#             return url

#         except ClientError as e:
#             raise Exception(f"Failed to generate presigned URL: {str(e)}")


# # Global storage service instance
# storage_service: Optional[StorageService] = None


# def get_storage_service() -> StorageService:
#     """Get or create storage service instance."""
#     global storage_service
#     if storage_service is None:
#         storage_service = StorageService()
#     return storage_service
