"""Embedding task for processing documents and storing vectors."""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from src.docarag.clients import get_minio_client, download_file_by_id
from src.docarag.services.parsers import parse_document
from src.docarag.services.embeddings import get_embedding_service
from src.docarag.services.vector_db import add_batch_objects
from src.docarag.settings import settings
from src.docarag.task_progress import _update_task_storage

logger = logging.getLogger(__name__)


async def run_embedding_task(task_id: str, document_id: str) -> None:
    """
    Background task to process document embeddings.

    Pipeline:
    1. Download file from MinIO
    2. Parse document into chunks
    3. Generate embeddings using gRPC service
    4. Store embeddings in vector database
    5. Update task status throughout

    Args:
        task_id: Unique task identifier
        document_id: Document ID to process
    """
    try:
        # Initialize task
        await _update_task_storage(
            task_id,
            file_id=document_id,
            status="processing",
            message="Starting embedding generation",
        )

        # Step 1: Download file from MinIO
        logger.info(f"Task {task_id}: Downloading file {document_id}")
        await _update_task_storage(
            task_id,
            message="Downloading file from storage",
        )

        client = get_minio_client()
        file_content, filename, metadata = download_file_by_id(
            client, settings.minio_bucket, document_id
        )
        content_type = metadata.get("content_type", "application/octet-stream")

        # Step 2: Parse document into chunks
        logger.info(f"Task {task_id}: Parsing document")
        await _update_task_storage(
            task_id,
            message="Parsing document into chunks",
        )

        chunks = parse_document(
            file_content=file_content,
            content_type=content_type,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            raise ValueError("No chunks extracted from document")

        logger.info(f"Task {task_id}: Parsed {len(chunks)} chunks")

        # Step 3: Generate embeddings
        logger.info(f"Task {task_id}: Generating embeddings for {len(chunks)} chunks")

        # Filter out empty or whitespace-only chunks
        valid_chunks = [
            chunk for chunk in chunks if chunk["content"] and chunk["content"].strip()
        ]

        if not valid_chunks:
            raise ValueError("No valid chunks with content found after filtering")

        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Task {task_id}: Filtered out {len(chunks) - len(valid_chunks)} empty chunks. "
                f"Processing {len(valid_chunks)} valid chunks."
            )

        # Set total chunks count at the beginning
        await _update_task_storage(
            task_id,
            message="Generating embeddings",
            total_chunks=len(valid_chunks),
        )

        embedding_service = get_embedding_service()
        texts = [chunk["content"] for chunk in valid_chunks]

        # Log detailed information about texts being sent
        logger.info(
            f"Task {task_id}: Generating embeddings for {len(texts)} texts. "
            f"Text lengths: min={min(len(t) for t in texts)}, "
            f"max={max(len(t) for t in texts)}, "
            f"avg={sum(len(t) for t in texts) / len(texts):.1f}"
        )

        # Process in batches to avoid timeouts
        batch_size = settings.embedding_batch_size
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            logger.info(
                f"Task {task_id}: Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} texts)"
            )

            batch_embeddings = await embedding_service.embed_batch_async(batch)
            embeddings.extend(batch_embeddings)

            # Update progress after each batch
            await _update_task_storage(
                task_id,
                message="Generating embeddings",
                chunks_processed=len(embeddings),
            )

            logger.info(
                f"Task {task_id}: Completed batch {batch_num}/{total_batches} "
                f"({len(embeddings)}/{len(texts)} total embeddings generated)"
            )

        logger.info(f"Task {task_id}: Generated {len(embeddings)} embeddings")

        # Step 4: Prepare batch objects for vector DB
        logger.info(f"Task {task_id}: Preparing batch objects for vector database")
        await _update_task_storage(
            task_id,
            message="Storing embeddings in vector database",
        )

        batch_objects: List[Dict[str, Any]] = []
        for chunk, embedding in zip(valid_chunks, embeddings):
            batch_objects.append(
                {
                    "properties": {
                        "document_name": filename,
                        "page": chunk["page"],
                        "content": chunk["content"],
                        "date_created": datetime.now(timezone.utc),
                    },
                    "vector": {
                        "content_vector": embedding,
                    },
                }
            )

        # Step 5: Store to vector database
        collection_name = "DefaultDocuments"
        await add_batch_objects(collection_name, batch_objects)

        logger.info(f"Task {task_id}: Successfully stored {len(batch_objects)} vectors")

        # Task completed successfully
        await _update_task_storage(
            task_id,
            status="completed",
            message=f"Successfully processed {len(valid_chunks)} chunks and stored embeddings",
            chunks_processed=len(valid_chunks),
            completed_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Task {task_id}: Failed with error: {str(e)}", exc_info=True)
        await _update_task_storage(
            task_id,
            status="failed",
            message=f"Failed to process embeddings: {str(e)}",
            completed_at=datetime.utcnow(),
        )
