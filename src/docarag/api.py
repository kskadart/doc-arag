from contextlib import asynccontextmanager
import logging
import uuid
import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends, status
from src.docarag.models import (
    ScrapeRequest,
    QueryRequest,
    UploadResponse,
    ScrapeResponse,
    EmbeddingResponse,
    QueryResponse,
    DeleteResponse,
    HealthResponse,
    Source,
    UploadedFileResponse,
    UploadedFilesListResponse,
)
from src.docarag.dependencies import upload_dependencies
from src.docarag.clients import check_vector_db_connection, get_minio_client, list_all_files, delete_file_by_id
from src.docarag.settings import settings
# from src.docarag.services.storage import get_storage_service
from src.docarag.services.uploader import process_upload
from src.docarag.services import (
    create_default_collection,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await check_vector_db_connection()
    await create_default_collection()
    # vector_db_service = get_vectorstore_service()
    # vector_db_service.create_schema(embedding_dimension=embedding_dim)

    # # Initialize embedding service (establishes gRPC connection)
    # embedding_service = get_embedding_service()
    # embedding_dim = await embedding_service.get_embedding_dimension_async()

    # # Initialize reranker service
    # # reranker_service = get_reranker_service()
    # # reranker_service.load_model()

    # # Initialize vector store schema
    # vectorstore = get_vectorstore_service()
    # vectorstore.create_schema(embedding_dimension=embedding_dim)

    # _ = get_rag_agent()

    yield

    # # Cleanup
    # vectorstore.close()
    # await embedding_service.close_async()


app = FastAPI(
    title="DOC ARAG API",
    description="Agentic RAG system for document processing and intelligent querying",
    version="0.1.0",
    lifespan=lifespan,
)


@app.delete("/documents/{file_id}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document(file_id: str):
    """
    Delete an uploaded file from MinIO storage by file_id.

    This will remove all files associated with the given file_id.
    """
    try:
        client = get_minio_client()
        all_files = list_all_files(client, settings.minio_bucket)
        
        file_exists = any(f["file_id"] == file_id for f in all_files)
        
        if not file_exists:
            raise HTTPException(
                status_code=404,
                detail=f"No file found with ID: {file_id}"
            )
        
        deleted_count = delete_file_by_id(client, settings.minio_bucket, file_id)
        
        return DeleteResponse(
            file_id=file_id,
            status="deleted",
            message=f"Successfully deleted {deleted_count} file(s) with ID: {file_id}",
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
        client = get_minio_client()
        all_files = list_all_files(client, settings.minio_bucket)
        
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


@app.post("/embeddings/{file_id}", response_model=EmbeddingResponse, tags=["Embedding"])
async def generate_embeddings(
    file_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Manually trigger embedding generation for an existing file in MinIO.

    This is useful for re-processing a document that was previously uploaded.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")
  

@app.get("/health", response_model=HealthResponse, tags=["Services"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", timestamp=datetime.datetime.now(datetime.UTC))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query the document collection using the RAG agent.

    The agent will:
    1. Understand and rephrase the query
    2. Retrieve relevant documents
    3. Rerank documents for relevance
    4. Generate an answer using Claude
    5. Evaluate and potentially iterate
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


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
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")



@app.get("/tasks/{task_id}", tags=["Tasks"])
async def get_task_status_endpoint(task_id: str):
    """
    Get the status of a background task.

    Useful for tracking upload, scraping, or embedding tasks.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@app.post("/uploads", response_model=UploadResponse, tags=["Uploads"])
async def upload_document_endpoint(
    upload_request=Depends(upload_dependencies),
):
    """
    Upload a document to the service storage.

    Supported file types: PDF, DOC, DOCX.
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
