from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from src.docarag.services.storage import get_storage_service

# from src.docarag.services.vectorstore import get_vectorstore_service
from src.docarag.services.embeddings import get_embedding_service
from src.docarag.services.parsers import parse_file, chunk_text
from src.docarag.services.scraper import scrape_url


# In-memory task status store (can be upgraded to Redis for production)
task_status: Dict[str, Dict[str, Any]] = {}


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get status of a background task.

    Args:
        task_id: Task identifier

    Returns:
        Task status dictionary or None if not found
    """
    return task_status.get(task_id)


def update_task_status(task_id: str, status: str, message: str, **kwargs) -> None:
    """
    Update task status.

    Args:
        task_id: Task identifier
        status: Status string (pending, processing, completed, failed)
        message: Status message
        **kwargs: Additional status fields
    """
    task_status[task_id] = {
        "task_id": task_id,
        "status": status,
        "message": message,
        "updated_at": datetime.utcnow().isoformat(),
        **kwargs,
    }


async def process_upload_task(
    task_id: str,
    file_id: str,
    filename: str,
    file_content: bytes,
    file_type: str,
    content_type: str,
) -> None:
    """
    Background task to process uploaded document.

    Args:
        task_id: Task identifier
        file_id: File identifier
        filename: Original filename
        file_content: File content as bytes
        file_type: File extension (pdf, docx)
        content_type: MIME type
    """
    try:
        update_task_status(task_id, "processing", f"Processing {filename}")

        # Get services
        storage = get_storage_service()
        # vectorstore = get_vectorstore_service()
        embeddings = get_embedding_service()

        # Step 1: Upload to S3
        update_task_status(task_id, "processing", "Uploading to storage")
        object_key = storage.upload_file(
            file_id=file_id,
            file_content=file_content,
            filename=filename,
            content_type=content_type,
            metadata={"type": file_type},
        )

        # Step 2: Parse document
        update_task_status(task_id, "processing", "Parsing document")
        text = parse_file(file_content, file_type)

        # Step 3: Chunk text
        update_task_status(task_id, "processing", "Chunking text")
        chunks = chunk_text(text)

        if not chunks:
            update_task_status(
                task_id, "failed", "No text content found in document", file_id=file_id
            )
            return

        # Step 4: Generate embeddings
        update_task_status(
            task_id, "processing", f"Generating embeddings for {len(chunks)} chunks"
        )
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_embeddings = embeddings.embed_batch(chunk_texts)

        # Step 5: Store vectors
        update_task_status(task_id, "processing", "Storing vectors")
        # vectorstore.add_vectors(
        #     file_id=file_id,
        #     chunks=chunks,
        #     embeddings=chunk_embeddings,
        #     source_type=file_type,
        #     filename=filename,
        #     metadata={"object_key": object_key}
        # )

        # Complete
        update_task_status(
            task_id,
            "completed",
            f"Successfully processed {filename}",
            file_id=file_id,
            chunks_count=len(chunks),
        )

    except Exception as e:
        update_task_status(task_id, "failed", f"Error: {str(e)}", file_id=file_id)


async def process_scraping_task(task_id: str, file_id: str, url: str) -> None:
    """
    Background task to scrape and process web page.

    Args:
        task_id: Task identifier
        file_id: File identifier
        url: URL to scrape
    """
    try:
        update_task_status(task_id, "processing", f"Scraping {url}")

        # Get services
        storage = get_storage_service()
        # vectorstore = get_vectorstore_service()
        embeddings = get_embedding_service()

        # Step 1: Scrape URL
        update_task_status(task_id, "processing", "Fetching web page")
        scraped_data = await scrape_url(url)

        html_content = scraped_data["html"]
        text_content = scraped_data["text"]
        title = scraped_data["title"]

        # Step 2: Save HTML to S3
        update_task_status(task_id, "processing", "Saving HTML to storage")
        filename = f"{title[:50]}.html".replace("/", "-")
        object_key = storage.upload_file(
            file_id=file_id,
            file_content=html_content.encode("utf-8"),
            filename=filename,
            content_type="text/html",
            metadata={"url": url, "title": title},
        )

        # Step 3: Chunk text
        update_task_status(task_id, "processing", "Chunking text")
        chunks = chunk_text(text_content)

        if not chunks:
            update_task_status(
                task_id,
                "failed",
                "No text content found on page",
                file_id=file_id,
                url=url,
            )
            return

        # Step 4: Generate embeddings
        update_task_status(
            task_id, "processing", f"Generating embeddings for {len(chunks)} chunks"
        )
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_embeddings = embeddings.embed_batch(chunk_texts)

        # Step 5: Store vectors
        update_task_status(task_id, "processing", "Storing vectors")
        # vectorstore.add_vectors(
        #     file_id=file_id,
        #     chunks=chunks,
        #     embeddings=chunk_embeddings,
        #     source_type="html",
        #     filename=filename,
        #     metadata={"url": url, "title": title, "object_key": object_key}
        # )

        # Complete
        update_task_status(
            task_id,
            "completed",
            f"Successfully processed {url}",
            file_id=file_id,
            url=url,
            chunks_count=len(chunks),
        )

    except Exception as e:
        update_task_status(
            task_id, "failed", f"Error: {str(e)}", file_id=file_id, url=url
        )


async def process_embedding_task(task_id: str, file_id: str) -> None:
    """
    Background task to generate embeddings for existing file.

    Args:
        task_id: Task identifier
        file_id: File identifier
    """
    try:
        update_task_status(
            task_id, "processing", f"Processing embeddings for {file_id}"
        )

        # Get services
        storage = get_storage_service()
        # vectorstore = get_vectorstore_service()
        embeddings = get_embedding_service()

        # Step 1: List files for this file_id
        update_task_status(task_id, "processing", "Retrieving file from storage")
        files = storage.list_files(prefix=file_id)

        if not files:
            update_task_status(
                task_id, "failed", f"No files found for {file_id}", file_id=file_id
            )
            return

        # Get first file
        file_obj = files[0]
        object_key = file_obj["key"]

        # Get file content and metadata
        file_content = storage.get_file(object_key)
        metadata = storage.get_file_metadata(object_key)

        # Determine file type from metadata or filename
        filename = object_key.split("/")[-1]
        file_type = filename.split(".")[-1].lower()

        # Step 2: Parse document
        update_task_status(task_id, "processing", "Parsing document")

        if file_type == "html":
            from src.docarag.services.scraper import clean_html_text

            text = clean_html_text(file_content.decode("utf-8"))
        else:
            text = parse_file(file_content, file_type)

        # Step 3: Chunk text
        update_task_status(task_id, "processing", "Chunking text")
        chunks = chunk_text(text)

        if not chunks:
            update_task_status(
                task_id, "failed", "No text content found", file_id=file_id
            )
            return

        # Step 4: Generate embeddings
        update_task_status(
            task_id, "processing", f"Generating embeddings for {len(chunks)} chunks"
        )
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_embeddings = embeddings.embed_batch(chunk_texts)

        # Step 5: Delete old vectors if any
        # vectorstore.delete_by_file_id(file_id)

        # # Step 6: Store new vectors
        # update_task_status(task_id, "processing", "Storing vectors")
        # vectorstore.add_vectors(
        #     file_id=file_id,
        #     chunks=chunks,
        #     embeddings=chunk_embeddings,
        #     source_type=file_type,
        #     filename=filename,
        #     metadata={"object_key": object_key}
        # )

        # Complete
        update_task_status(
            task_id,
            "completed",
            f"Successfully processed embeddings for {file_id}",
            file_id=file_id,
            chunks_count=len(chunks),
        )

    except Exception as e:
        update_task_status(task_id, "failed", f"Error: {str(e)}", file_id=file_id)


def create_task_id() -> str:
    """Generate a unique task ID."""
    return str(uuid.uuid4())
