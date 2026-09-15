from contextlib import asynccontextmanager
import logging
import datetime
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends, status
from src.docarag.models import (
    ScrapeRequest,
    QueryRequest,
    UploadResponse,
    ScrapeResponse,
    EmbeddingResponse,
    AgentQueryResponse,
    DeleteResponse,
    HealthResponse,
    MeResponse,
    UploadedFileResponse,
    UploadedFilesListResponse,
    TaskStatusResponse,
)
from src.docarag.dependencies import upload_dependencies, get_all_files
from src.docarag.auth import CurrentUser, get_current_user, require_admin
from src.docarag.clients import (
    check_vector_db_connection,
    get_minio_client,
    delete_file_by_id,
)
from src.docarag.settings import settings

# from src.docarag.services.storage import get_storage_service
from src.docarag.services import (
    process_upload,
    create_default_collection,
    delete_objects_by_document_name,
    verify_embedding_dimension,
)
from src.docarag.consts import DEFAULT_COLLECTION_NAME
from src.docarag.tasks import run_embedding_task
from src.docarag.task_progress import get_task


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await check_vector_db_connection()
    await create_default_collection()
    if settings.startup_verify_embedding_dimension:
        await verify_embedding_dimension()

    yield


app = FastAPI(
    title="DOC ARAG API",
    description="Agentic RAG system for document processing and intelligent querying",
    version="0.1.0",
    lifespan=lifespan,
)


@app.delete(
    "/documents/{document_id}", response_model=DeleteResponse, tags=["Documents"]
)
async def delete_document(
    document_id: str,
    _admin: CurrentUser = Depends(require_admin),
    all_files: list[dict] = Depends(get_all_files),
):
    """
    Delete an uploaded file from MinIO storage and the vector database.

    This will remove all files associated with the given document_id together
    with every chunk embedded from them.
    """
    document_files = [f for f in all_files if f["file_id"] == document_id]
    if not document_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No file found with ID: {document_id}",
        )
    try:
        client = get_minio_client()
        deleted_count = delete_file_by_id(client, settings.minio_bucket, document_id)

        deleted_chunks = 0
        for file_info in document_files:
            deleted_chunks += await delete_objects_by_document_name(
                DEFAULT_COLLECTION_NAME, file_info["filename"]
            )

        return DeleteResponse(
            file_id=document_id,
            status="deleted",
            message=(
                f"Successfully deleted {deleted_count} file(s) "
                f"and {deleted_chunks} chunk(s) with ID: {document_id}"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting uploaded file: {str(e)}",
        )


@app.get("/documents", response_model=UploadedFilesListResponse, tags=["Documents"])
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    _admin: CurrentUser = Depends(require_admin),
    all_files: list[dict] = Depends(get_all_files),
):
    """
    List all uploaded files with metadata from MinIO storage.

    Returns paginated list of files with their metadata including:
    - File ID
    - Filename
    - Size
    - Content type
    - Upload timestamp
    - Custom metadata
    """
    try:
        total = len(all_files)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_files = all_files[start_idx:end_idx]

        files = [
            UploadedFileResponse(
                file_id=file_info["file_id"],
                object_key=file_info["object_key"],
                filename=file_info["filename"],
                size_bytes=file_info["size_bytes"],
                content_type=file_info["content_type"],
                last_modified=file_info["last_modified"],
                metadata=file_info["metadata"],
            )
            for file_info in paginated_files
        ]

        return UploadedFilesListResponse(
            files=files,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing uploaded files: {str(e)}",
        )


@app.post(
    "/embeddings/{document_id}", response_model=EmbeddingResponse, tags=["Embedding"]
)
async def generate_embeddings(
    document_id: str,
    background_tasks: BackgroundTasks,
    _admin: CurrentUser = Depends(require_admin),
    all_files: list[dict] = Depends(get_all_files),
):
    """
    Start background task to generate embeddings for a document.

    Args:
        document_id: ID of the document to process
        background_tasks: Background task manager
        all_files: List of all files for validation

    Returns:
        EmbeddingResponse with task_id and processing status
    """
    # Validate document exists
    file_exists = any(f["file_id"] == document_id for f in all_files)
    if not file_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No file found with ID: {document_id}",
        )

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    background_tasks.add_task(run_embedding_task, task_id, document_id)

    return EmbeddingResponse(
        task_id=task_id,
        file_id=document_id,
        status="processing",
        message=f"Embedding generation started. Use task_id to check progress at /tasks/{task_id}",
    )


@app.get("/health", response_model=HealthResponse, tags=["Services"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.datetime.now(datetime.UTC),
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model
        if settings.llm_provider == "openai"
        else settings.anthropic_model,
        embedding_model=settings.embedding_model,
        reranker_provider=settings.reranker_provider,
    )


@app.get("/me", response_model=MeResponse, tags=["Services"])
async def me(user: CurrentUser = Depends(get_current_user)):
    """The caller as asserted by the edge proxy; lets the UI adapt to the role."""
    return MeResponse(
        username=user.username,
        display_name=user.display_name,
        email=user.email,
        groups=sorted(user.groups),
        is_admin=user.is_admin,
        auth_mode=settings.auth_mode,
    )


@app.post("/query", response_model=AgentQueryResponse, tags=["Query"])
async def query_documents_endpoint(
    request: QueryRequest, user: CurrentUser = Depends(get_current_user)
):
    """
    Query the document collection using the RAG agent.

    The agent will:
    1. Understand and rephrase the query
    2. Retrieve relevant documents using vector search
    3. Generate an answer using the configured LLM
    4. Evaluate and potentially iterate

    Returns the agent's generated answer with confidence score and metadata.
    """
    from src.docarag.services.agent import query_documents

    logger.debug("query from %s", user.username)
    try:
        return await query_documents(request)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )


@app.post("/scrappings", response_model=ScrapeResponse, tags=["Documents"])
async def scrape_webpage(
    background_tasks: BackgroundTasks,
    request: ScrapeRequest,
):
    """
    Scrape a web page and process it.

    The web page will be:
    1. Scraped and HTML saved to S3
    2. Text extracted and chunked
    3. Embedded using the embedding model
    4. Stored in the vector database
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented"
    )


@app.get("/tasks/{task_id}", tags=["Tasks"])
async def get_task_status_endpoint(
    task_id: str, _user: CurrentUser = Depends(get_current_user)
):
    """
    Get the status of a background task.

    Useful for tracking upload, scraping, or embedding tasks.

    Args:
        task_id: Unique task identifier

    Returns:
        TaskStatusResponse with current task status

    Raises:
        HTTPException: 404 if task not found
    """
    task = await get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )

    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        file_id=task.get("file_id"),
        message=task["message"],
        chunks_processed=task.get("chunks_processed", 0),
        total_chunks=task.get("total_chunks", 0),
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
    )


@app.post("/uploads", response_model=UploadResponse, tags=["Uploads"])
async def upload_document_endpoint(
    # The guard comes first: the multipart body is not parsed for non-admins
    _admin: CurrentUser = Depends(require_admin),
    upload_request=Depends(upload_dependencies),
):
    """
    Upload a document to the service storage.

    Supported file types: PDF, DOC, DOCX, MD.
    """
    try:
        upload_result = await process_upload(upload_request)

        return UploadResponse(
            file_id=upload_result["file_id"],
            filename=upload_result["filename"],
            status="completed",
            message=f"Document uploaded successfully to MinIO at {upload_result['object_key']}",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}",
        )
