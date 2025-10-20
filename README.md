# This is DOC-ARAG: Agentic RAG System

An intelligent document processing and retrieval system using LangGraph, Weaviate, and Claude AI.

## Features

- **Document Processing**: Parse PDF and DOCX files
- **Web Scraping**: Extract and index content from web pages
- **Vector Search**: Store and retrieve document embeddings using Weaviate
- **Local Embeddings**: Use sentence-transformers for embedding generation
- **Reranking**: Improve search results with cross-encoder reranking
- **Agentic RAG**: Intelligent query processing with LangGraph agent
- **Background Processing**: Async document processing with FastAPI background tasks
- **S3 Storage**: Store original documents in MinIO (S3-compatible)

## Architecture

### Components

- **FastAPI**: REST API server
- **MinIO**: S3-compatible object storage for documents
- **Weaviate**: Vector database for embeddings
- **EmbeddingGemma (300M)**: Local embedding generation with Google's EmbeddingGemma
- **LangGraph**: Agent workflow orchestration
- **Claude-3-Haiku**: LLM for answer generation
- **Cross-Encoder**: Document reranking

### Agent Workflow

1. **Understand Query**: Analyze and rephrase user query
2. **Retrieve**: Get top-k candidates from vector store
3. **Rerank**: Apply cross-encoder for better relevance
4. **Generate**: Create answer using Claude with context
5. **Evaluate**: Assess quality and decide to iterate or finish

## Prerequisites

- Docker and Docker Compose
- Python 3.13+
- uv (Python package manager)

## Installation

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set environment variables**:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - Other settings are configured in `compose.yml` for Docker deployment

## Running the Application

### Using Docker Compose (Recommended)

```bash
docker compose up --build
```

This will start:
- FastAPI application on `http://localhost:8103`
- MinIO console on `http://localhost:9001` (user: minioadmin, pass: minioadmin)
- Weaviate on `http://localhost:8080`

### Local Development

```bash
# Ensure services are running (MinIO, Weaviate)
docker compose up minio weaviate -d

# Run the application
uv run fastapi dev src/docarag/main.py --host 0.0.0.0 --port 8103
```

## API Endpoints

### Health Check
```http
GET /health
```

### Upload Document
```http
POST /uploads
Content-Type: multipart/form-data

file: <PDF or DOCX file>
```

### Scrape Web Page
```http
POST /scrappings
Content-Type: application/json

{
  "url": "https://example.com/article"
}
```

### Generate Embeddings
```http
POST /embeddings/{file_id}
```

### Query Documents
```http
POST /query
Content-Type: application/json

{
  "query": "What is the main topic?",
  "file_id": "optional-file-id",
  "source_type": "pdf|docx|html",
  "max_iterations": 2
}
```

### List Documents
```http
GET /documents?page=1&page_size=10
```

### Delete Document
```http
DELETE /documents/{file_id}
```

### Get Task Status
```http
GET /tasks/{task_id}
```

## Configuration

Settings are managed in `src/docarag/config.py` using Pydantic Settings.

Key configuration options:
- `ANTHROPIC_API_KEY`: Claude API key
- `ANTHROPIC_MODEL`: Model name (default: claude-3-haiku-20240307)
- `EMBEDDING_MODEL_NAME`: EmbeddingGemma model (default: google/embeddinggemma-300m)
- `RERANKER_MODEL_NAME`: Cross-encoder model
- `CHUNK_SIZE`: Text chunk size (default: 512)
- `CHUNK_OVERLAP`: Chunk overlap (default: 50)
- `INITIAL_RETRIEVAL_K`: Initial retrieval count (default: 20)
- `RERANK_TOP_K`: Final reranked results (default: 5)

## Testing

Run tests:
```bash
uv run pytest tests/
```

Run with coverage:
```bash
uv run pytest tests/ --cov=src/docarag
```

## Project Structure

```
src/docarag/
├── api.py              # FastAPI endpoints
├── main.py             # Entry point
├── config.py           # Configuration
├── models/             # Pydantic models
│   ├── requests.py
│   └── responses.py
├── services/           # Core services
│   ├── storage.py      # MinIO S3 client
│   ├── vectorstore.py  # Weaviate client
│   ├── embeddings.py   # Embedding generation
│   ├── reranker.py     # Document reranking
│   ├── parsers.py      # PDF/DOCX parsing
│   ├── scraper.py      # Web scraping
│   └── rag_agent.py    # LangGraph agent
└── utils/              # Utilities
    └── background_tasks.py  # Background processing
```

## Development

### Code Quality

Format code:
```bash
uv run black src/ tests/
```

Lint code:
```bash
uv run ruff check src/ tests/
```

Type check:
```bash
uv run mypy src/
```

## Usage Examples

### 1. Upload a PDF Document

```bash
curl -X POST "http://localhost:8103/uploads" \
  -F "file=@document.pdf"
```

### 2. Scrape a Web Page

```bash
curl -X POST "http://localhost:8103/scrappings" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

### 3. Query the Knowledge Base

```bash
curl -X POST "http://localhost:8103/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "max_iterations": 2
  }'
```

### 4. Check Task Status

```bash
curl -X GET "http://localhost:8103/tasks/{task_id}"
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation as needed
4. Use absolute imports: `from src.docarag.services import ...`

## License

MIT
