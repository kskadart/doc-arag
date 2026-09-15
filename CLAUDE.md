# doc-arag — правила для Claude Code

Agentic RAG-сервис (FastAPI + LangGraph + Weaviate + MinIO). Проект OREO: ассистент поддержки операторов «Эра-Телеком» поверх корпуса `oreo-data`.

## Кластер репозиториев (`/Users/kskada/develop/`)
- `doc-arag` — этот бэкенд (:8103). Владеет docker-сетью `arag-common-network`, поднимается первым.
- `oreo-data` — корпус: `corpus/<domain>/*.md` (14 файлов, 9 доменов). Не грузить `corpus/EXCLUSIONS.md`, `corpus/fio-blacklist.txt`, `corpus/_raw/`. Golden set для оценки качества — `golden/` (схема в `golden/README.md`; `must_include` обязаны дословно быть в источнике).
- `rag-services` — локальные gRPC-модели на CPU (embedding :8351, reranker :8352, compose-имя сервиса ровно `reranker-service`).
- `doc-arag-client` — Next.js UI; шлёт `domain: "DefaultDocuments"` в `/query` (= «без фильтра»).

## Где мы и почему
- `.claude/progress.md` — текущий чек-лист этапов и статус. **Читать первым, обновлять после каждого этапа.**
- `.claude/decisions.md` — принятые решения (ADR-lite).
- `.claude/backlog.md` — известные проблемы вне текущего скоупа.
- План и контрольные вопросы: `oreo-data/docs/plan.md` (§3.2 — 12 вопросов).

## Команды
```bash
uv sync                          # окружение (python из .python-version)
make tests linter typecheck      # обязательно зелёные перед любым коммитом
make formatter                   # black + ruff format (CI проверяет только ruff format)
docker compose up -d             # api + weaviate + minio (модели по API — OpenRouter)
scripts/local_models.sh start|status|stop   # llama.cpp нативно на Metal (:8081 chat, :8082 embed, :8083 rerank)
docker compose -f compose.yml -f compose.models.yml up -d   # api → host.docker.internal (локальные модели)
docker compose -f compose.yml -f compose.sglang.yml up -d   # Linux+NVIDIA: SGLang ×3 (на Mac не работает)
uv run python -m scripts.load_corpus --dry-run   # отчёт по чанкам без docker и ключей
uv run python -m scripts.load_corpus             # загрузка корпуса в стек
uv run python -m scripts.run_control_questions   # прогон 12 контрольных вопросов
uv run python -m scripts.eval_golden --validate  # golden set: parts/*.jsonl → golden-set.jsonl + проверка по корпусу
uv run python -m scripts.eval_golden [--judge]   # оценка сервиса на golden set
```
Системные зависимости на macOS: `brew install uv libmagic`; Docker Desktop должен быть запущен.

## Жёсткие правила
- **Git:** commit, push, merge — только по явной команде пользователя. Push в `main` запускает deploy.yml (прод на Yandex Cloud сейчас выключен, но workflow жив).
- **Версии:** только статические пины `pkg==X.Y.Z` в `pyproject.toml`; после `uv add` — перепинить точную версию.
- **Settings (`src/docarag/settings.py`) — `extra="forbid"`:** любой неизвестный ключ в `.env` роняет старт. Новые параметры — только полями с дефолтом в коде. `.env` писать по `settings.py`; `env.example` — образец, не копировать вслепую.
- **Имена документов и файлов корпуса — только латиница** (кириллица ломает S3-метаданные MinIO). `domain` — slug `^[a-z0-9][a-z0-9-]*$`.
- **Модели — через env, не через код:** `OPENROUTER_API_KEY` + `LLM_*`, `EMBEDDING_*`, `RERANKER_*`. По умолчанию OpenRouter: `qwen/qwen3.8-flash`, `qwen/qwen3-embedding-8b`, `qwen/qwen3-reranker-8b`; локально на Mac — `scripts/local_models.sh` (llama.cpp нативно, Metal) + `compose.models.yml`; на GPU-сервере — `compose.sglang.yml`. **Docker Desktop на Mac не отдаёт GPU контейнерам** (только Windows/WSL2), SGLang-образы CUDA-only — не пытаться поднимать SGLang в docker на Mac. Смена эмбеддера = `load_corpus --recreate` (размерность вектора меняется).
- **Сабагенты:** не на Fable; `model: "sonnet"` для простых задач, `model: "opus"` для сложных. Всегда передавать `model` явно.
- **Корпус:** плейсхолдеры `{{сумма}}`, `{{имя_абонента}}` и условные названия («ООО Ромашка», ID 12345) — намеренные, не «чинить». «Эр-Телеком Холдинг» и ELAN-адреса — реальные, утверждены.
- **Black-долг** (9 файлов) не трогать: CI гоняет только `ruff format --check`.
- pb2-стабы регенерировать только пиненой командой из README (`--with grpcio-tools==1.78.0 protobuf==6.33.6`).

## Архитектура в двух строках
Upload: `POST /uploads` (multipart: `document_name`, `domain`, файл) → MinIO → `POST /embeddings/{id}` → фоновая задача (parse → embed → purge-before-insert → Weaviate `DefaultDocuments`, named vector `content_vector`) → `GET /tasks/{id}`.
Query: `POST /query` → LangGraph: rephrase → embed → retrieve (k=20, опц. фильтр `domain`) → rerank (graceful fallback) → generate → evaluate (порог 0.7).
Auth: бэкенд паролей не знает — Caddy+Authelia (`doc-arag-client`) кладут `Remote-User`/`Remote-Groups`, `src/docarag/auth.py` их читает при `AUTH_MODE=trusted-headers` (по умолчанию `none` = аноним-админ). Порты api/weaviate/minio — только `127.0.0.1`, не открывать.
