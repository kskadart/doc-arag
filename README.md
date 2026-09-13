# doc-arag — Agentic RAG Service

Document ingestion and question answering over a private corpus: FastAPI + LangGraph + Weaviate + MinIO. Models are pluggable per component through environment variables: chat, embeddings and rerank all speak the OpenAI / Cohere-style HTTP contracts, so the same code runs against OpenRouter (default: `qwen/qwen3.8-flash`, `qwen/qwen3-embedding-8b`, `qwen/qwen3-reranker-8b`) or a local llama.cpp stack (`compose.models.yml`) with the open Qwen weights.

## Architecture

- **FastAPI** (`src/docarag/api.py`) — REST API on `:8103`.
- **MinIO** — stores the original documents (`file_id/filename`, `domain` in object metadata).
- **Weaviate 1.39** — one collection `DefaultDocuments`, named vector `content_vector`, properties `document_name`, `page`, `content`, `domain`, `date_created`.
- **LangGraph agent** (`services/agent.py`) — rephrase → embed → retrieve (k=20, optional `domain` filter) → rerank (graceful fallback) → generate → evaluate (threshold 0.7, up to `max_iterations`).
- **Providers** — `services/llm.py` (chat), `clients/embedding_http.py` (embeddings), `clients/reranker_http.py` / `clients/reranker_client.py` (reranker), all selected in `settings.py`.

### Ingestion pipeline

1. `POST /uploads` (multipart: `document_name`, `domain`, `document` file) → MinIO. Supported: PDF, Markdown (`text/markdown` / `text/plain`); DOCX is accepted but not parsed yet.
2. `POST /embeddings/{file_id}` → background task: parse → chunk → embed → purge previous chunks of the same `document_name` → `insert_many` into Weaviate.
3. `GET /tasks/{task_id}` → `processing` / `completed` / `failed`.

Markdown is split by headers (`#`, `##`, `###`), each chunk carries a `H1 > H2 > H3` breadcrumb, YAML front-matter is dropped, `page` is the section ordinal. Oversized sections are sub-split at `MD_CHUNK_SIZE` (1800 chars).

## Prerequisites

- Docker Desktop (Weaviate + MinIO run in compose)
- Python 3.13 and `uv` (`brew install uv`)
- `libmagic` for MIME detection (`brew install libmagic` on macOS)
- Either an OpenRouter key (`OPENROUTER_API_KEY`, used for chat, embeddings and rerank) or the local model stack below

## Setup

```bash
uv sync
cp env.example .env    # then edit: keep ONLY keys that exist in src/docarag/settings.py
docker compose up -d   # api + weaviate + minio
curl http://localhost:8103/health
```

`Settings` is `extra="forbid"`: an unknown key in `.env` aborts startup. The compose-only keys at the top of `env.example` (`MINIO_ROOT_*`) must not be in the application `.env`.

### Choosing models

| Component | Setting | Default | Alternatives |
|---|---|---|---|
| Chat | `LLM_PROVIDER=openai`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenRouter, `qwen/qwen3.8-flash` | local llama.cpp / Ollama / vLLM, `LLM_PROVIDER=anthropic` |
| Embeddings | `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` | OpenRouter, `qwen/qwen3-embedding-8b` (4096 dims) | any `/v1/embeddings` server; `EMBEDDING_DIMENSIONS` when the model supports it |
| Reranker | `RERANKER_PROVIDER=openai-rerank\|grpc\|none` | OpenRouter, `qwen/qwen3-reranker-8b` | llama.cpp / vLLM / Jina `/v1/rerank`, rag-services gRPC, or `none` |

`OPENROUTER_API_KEY` fills the key of every component whose base URL is OpenRouter; a component-specific `*_API_KEY` overrides it.

### Local model stack (macOS, llama.cpp on Metal)

Docker Desktop cannot expose the Apple GPU to Linux containers (Docker's GPU support exists only on Windows/WSL2), so on a Mac the model servers run natively and the api container reaches them through `host.docker.internal`:

```bash
brew install llama.cpp
scripts/local_models.sh start        # chat Qwen3.8-27B :8081, Qwen3-Embedding-8B :8082, Qwen3-Reranker-8B :8083 (weights pulled from Hugging Face into ~/.cache/huggingface/hub)
scripts/local_models.sh status
docker compose -f compose.yml -f compose.models.yml up -d
```

Same OpenAI / `/v1/rerank` contracts as OpenRouter, so `.env` is the only difference between local testing and API deployment. Override weights with `LLM_HF_REPO`, `EMBEDDING_HF_REPO`, `RERANKER_HF_REPO` (`repo:quant`). Qwen3.8-Flash is closed-weight; Qwen3.8-27B is the open counterpart.

### GPU server stack (Linux + NVIDIA, SGLang)

```bash
docker compose -f compose.yml -f compose.sglang.yml up -d
```

Three `lmsysorg/sglang` containers (chat, `--is-embedding` embeddings, decoder-only Qwen3 reranker with its yes/no chat template). SGLang's `/v1/rerank` answers with a bare list of `{index, score}`; the reranker client accepts that dialect as well as the `results[{index, relevance_score}]` one.

## Loading a corpus

```bash
uv run python -m scripts.load_corpus --dry-run                 # chunk report, no services or keys needed
uv run python -m scripts.load_corpus --recreate --only diagnostics/internet-check-procedure.md
uv run python -m scripts.load_corpus                           # whole corpus, skips already uploaded files
uv run python -m scripts.load_corpus --force                   # delete + re-upload
```

The loader walks `<corpus>/<domain>/*.md`, sends `domain` = directory name, waits for each embedding task and fails unless the Weaviate object count grows. Directories starting with `_` and `EXCLUSIONS.md` are skipped; file names must be latin.

## Evaluating models

```bash
uv run python -m scripts.run_control_questions                 # 12 built-in questions → .claude/reports/
uv run python -m scripts.run_control_questions --questions my.json --use-domain-filter
```

The report stores the configuration reported by `GET /health` (`llm_model`, `embedding_model`, `reranker_provider`), so runs with different `.env` values can be compared. Comparing chat models: change `LLM_MODEL`, restart the api, rerun. Comparing embedders: also `load_corpus --recreate`.

### Golden set

The golden set lives next to the corpus (`oreo-data/golden/`, schema in its README): question, reference answer, source file and section, `must_include` facts that are verified to exist in the source, `must_not_include` strings, record type (factual / procedural / paraphrase / cross-doc / negative).

```bash
uv run python -m scripts.eval_golden --validate                        # parts/*.jsonl → golden-set.jsonl, checked against the corpus
uv run python -m scripts.eval_golden                                   # doc_hit@k, domain_hit@k, fact coverage, forbidden strings
uv run python -m scripts.eval_golden --domain diagnostics --use-domain-filter
uv run python -m scripts.eval_golden --judge --judge-model qwen/qwen3.7-plus   # + LLM grade 1-5 against the reference
```

`POST /query` returns the chunks used as context in `sources` (document, domain, section ordinal, score, snippet), which is what the retrieval metrics are computed from.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | status + configured models |
| `POST` | `/uploads` | multipart upload (`document_name`, `domain`, `document` or `document_url`) |
| `POST` | `/embeddings/{file_id}` | start the embedding task |
| `GET` | `/tasks/{task_id}` | task progress |
| `GET` | `/documents?page=&page_size=` | list uploaded documents |
| `DELETE` | `/documents/{file_id}` | delete from MinIO and Weaviate |
| `POST` | `/query` | `{"query": "...", "domain": "diagnostics" \| null, "max_iterations": 2}` |

`domain` in `/query` is an optional chunk filter; an empty value or the legacy collection name `DefaultDocuments` means no filter.

```bash
curl -X POST http://localhost:8103/uploads \
  -F "document_name=internet-check-procedure.md" -F "domain=diagnostics" \
  -F "document=@corpus/diagnostics/internet-check-procedure.md;type=text/markdown"

curl -X POST http://localhost:8103/query -H "Content-Type: application/json" \
  -d '{"query": "У абонента не работает интернет — какие шаги проверки нужно выполнить?"}'
```

## Development

```bash
make tests linter typecheck   # pytest, ruff, mypy (src/ tests/ scripts/)
make formatter                # black + ruff format (CI checks ruff format only)
make corpus-dry-run
```

gRPC stubs for the reranker are committed; regenerate only with the pinned toolchain:

```bash
uv run --with grpcio-tools==1.78.0 --with protobuf==6.33.6 \
  python -m grpc_tools.protoc -I proto --python_out=src/docarag --grpc_python_out=src/docarag proto/reranker.proto
```

## Project structure

```
src/docarag/
├── api.py                  # FastAPI routes and lifespan
├── settings.py             # pydantic-settings, extra="forbid"
├── consts.py, errors.py
├── clients/
│   ├── embedding_http.py   # OpenAI-compatible /embeddings
│   ├── reranker_http.py    # /rerank (TEI, vLLM, Jina)
│   ├── reranker_client.py  # gRPC reranker (rag-services)
│   ├── minio_client.py, vector_db_client.py
├── services/
│   ├── agent.py            # LangGraph workflow
│   ├── llm.py              # chat model factory
│   ├── embeddings.py, reranker.py
│   ├── parsers.py          # PDF and Markdown chunking
│   ├── uploader.py, vector_db.py
├── tasks/embedding_task.py
└── models/                 # request / response schemas
scripts/
├── load_corpus.py
├── run_control_questions.py
└── eval_golden.py
```

Project conventions for Claude Code live in `CLAUDE.md`; progress, decisions and backlog in `.claude/`.
