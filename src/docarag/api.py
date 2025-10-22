from contextlib import asynccontextmanager
import logging
import uuid
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from src.docarag.settings import settings
from src.docarag.models import (
    ScrapeRequest,
    QueryRequest,
    UploadResponse,
    ScrapeResponse,
    EmbeddingResponse,
    QueryResponse,
    DocumentListResponse,
    DeleteResponse,
    HealthResponse,
    DocumentResponse,
    Source,
)
from src.docarag.clients import check_vector_db_connection
from src.docarag.services.rag_agent import get_rag_agent
from src.docarag.services.storage import get_storage_service
from src.docarag.services import (
    create_default_collection,
)
from src.docarag.utils.background_tasks import (
    process_upload_task,
    process_scraping_task,
    process_embedding_task,
    create_task_id,
    get_task_status,
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", timestamp=datetime.datetime.now(datetime.UTC))


@app.post("/uploads", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document (PDF or DOCX) for processing.

    The document will be:
    1. Uploaded to S3 storage
    2. Parsed and chunked
    3. Embedded using the embedding model
    4. Stored in the vector database
    """
    # Validate file type
    filename = file.filename or "unknown"
    file_ext = filename.split(".")[-1].lower()

    if file_ext not in ["pdf", "docx", "doc"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Only PDF and DOCX are supported.",
        )

    # Read file content
    file_content = await file.read()

    # Validate file size
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size_mb:.2f}MB. Maximum size is {settings.max_file_size_mb}MB.",
        )

    # Generate IDs
    file_id = str(uuid.uuid4())
    task_id = create_task_id()

    # Determine content type
    content_type = (
        "application/pdf"
        if file_ext == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    # Start background processing
    background_tasks.add_task(
        process_upload_task,
        task_id=task_id,
        file_id=file_id,
        filename=filename,
        file_content=file_content,
        file_type=file_ext,
        content_type=content_type,
    )

    return UploadResponse(
        file_id=file_id,
        filename=filename,
        status="processing",
        message=f"Document upload initiated. Task ID: {task_id}",
    )


@app.post("/scrappings", response_model=ScrapeResponse)
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
    url = str(request.url)

    # Generate IDs
    file_id = str(uuid.uuid4())
    task_id = create_task_id()

    # Start background processing
    background_tasks.add_task(
        process_scraping_task,
        task_id=task_id,
        file_id=file_id,
        url=url,
    )

    return ScrapeResponse(
        file_id=file_id,
        url=url,
        status="processing",
        message=f"Web scraping initiated. Task ID: {task_id}",
    )


@app.post("/embeddings/{file_id}", response_model=EmbeddingResponse)
async def generate_embeddings(
    file_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Manually trigger embedding generation for an existing file in S3.

    This is useful for re-processing a document that was previously uploaded.
    """
    # Verify file exists
    storage = get_storage_service()
    files = storage.list_files(prefix=file_id)

    if not files:
        raise HTTPException(status_code=404, detail=f"No file found with ID: {file_id}")

    # Generate task ID
    task_id = create_task_id()

    # Start background processing
    background_tasks.add_task(
        process_embedding_task,
        task_id=task_id,
        file_id=file_id,
    )

    return EmbeddingResponse(
        file_id=file_id,
        status="processing",
        message=f"Embedding generation initiated. Task ID: {task_id}",
    )


@app.post("/query", response_model=QueryResponse)
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
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Get RAG agent
        agent = get_rag_agent()

        # Invoke agent
        result = await agent.ainvoke(
            query=request.query,
            file_id=request.file_id,
            source_type=request.source_type,
            max_iterations=request.max_iterations,
        )

        # Format response
        return QueryResponse(
            answer=result["answer"],
            sources=[
                Source(
                    file_id=src["file_id"],
                    content=src["content"],
                    score=src["score"],
                    source_type=src["source_type"],
                    chunk_index=src["chunk_index"],
                )
                for src in result["sources"]
            ],
            confidence=result["confidence"],
            iterations=result["iterations"],
            rephrased_query=result.get("rephrased_query"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
):
    """
    List all indexed documents with metadata.

    Supports pagination.
    """
    try:
        vectorstore = get_vectorstore_service()
        storage = get_storage_service()

        # Get documents metadata from vector store
        docs_metadata = vectorstore.get_documents_metadata(limit=1000)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = docs_metadata[start_idx:end_idx]

        # Enrich with storage metadata
        documents = []
        for doc in paginated_docs:
            try:
                # Get file size from storage
                files = storage.list_files(prefix=doc["file_id"])
                size_bytes = files[0]["size"] if files else 0
                created_at = files[0]["last_modified"] if files else datetime.utcnow()

                documents.append(
                    DocumentResponse(
                        file_id=doc["file_id"],
                        filename=doc["filename"],
                        source_type=doc["source_type"],
                        size_bytes=size_bytes,
                        created_at=created_at,
                        chunks_count=doc["chunks_count"],
                    )
                )
            except Exception:
                # Skip documents that can't be accessed
                continue

        return DocumentListResponse(
            documents=documents,
            total=len(docs_metadata),
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing documents: {str(e)}"
        )


@app.delete("/documents/{file_id}", response_model=DeleteResponse)
async def delete_document(file_id: str):
    """
    Delete a document from both S3 storage and vector database.
    """
    try:
        vectorstore = get_vectorstore_service()
        storage = get_storage_service()

        # Delete from vector store
        deleted_vectors = vectorstore.delete_by_file_id(file_id)

        # Delete from S3
        deleted_files = storage.delete_by_prefix(file_id)

        if deleted_vectors == 0 and deleted_files == 0:
            raise HTTPException(
                status_code=404, detail=f"No document found with ID: {file_id}"
            )

        return DeleteResponse(
            file_id=file_id,
            status="deleted",
            message=f"Deleted {deleted_files} file(s) and {deleted_vectors} vector(s)",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )


@app.get("/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """
    Get the status of a background task.

    Useful for tracking upload, scraping, or embedding tasks.
    """
    status = get_task_status(task_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return status
