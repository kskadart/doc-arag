# Backlog (вне текущего скоупа; из аудита 2026-09-13)

## Агент
- Итерации бесполезны: `rephrase_query_node` всегда читает `state.query`, второй круг повторяет первый (3 лишних LLM-вызова). Передавать в rephrase историю/предыдущий rephrased_query или убирать цикл.
- Промпты inline f-string без system message; вынести в шаблоны.

## Эксплуатация
- In-memory task store (`task_progress.py`) + `--workers 2` в `docker/Dockerfile.prod` → `GET /tasks/{id}` 404 на «чужом» воркере; нет вытеснения записей.
- Weaviate `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true` и MinIO дефолтный пароль в `docker/compose.prod.yml`; нет auth/rate-limit на API.
- Прод-secrets для новых `LLM_*/EMBEDDING_*/RERANKER_*` ключей в `deploy.yml` — когда прод оживёт.
- `datetime.utcnow()` (deprecated) в `responses.py`, `task_progress.py`, `embedding_task.py`.
- `minio_client.py`: `urlparse("host:port")` даёт scheme=host → `secure` всегда False для bare endpoint.

## Мёртвый код
- `services/storage.py`, `services/scraper.py` + `POST /scrappings` (501), `utils/default_collection_conf.py: DEFAULT_COLLECTION_CONFIG` + `create_collection_from_config` (сломанный subscript), `dependencies.py: file_downloader / parse_document_dependency`, `models/responses.py: Source, QueryResponse, DocumentResponse, DocumentListResponse`.
- `tests/conftest.py` стабит несуществующие `services.rag_agent` и `utils.background_tasks`; `tests/test_api.py` — два skip с устаревшей причиной, upload-тест шлёт неверное имя поля.
- Зависимости без импортов: `langchain-community`, `python-docx`, `beautifulsoup4`.

## Инфраструктура
- `docker/compose.prod.yml` и `deploy.yml` получили новые `LLM_*/EMBEDDING_*/RERANKER_*` ключи, но GitHub secrets/vars под них не заведены — прод выключен, сделать при оживлении.
- Локальный чат Qwen3.8-27B: без `--reasoning off` весь `max_tokens` уходит в thinking и `content` пустой; для других серверов (vLLM/SGLang) держать `LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}` (SGLang: `--reasoning-parser qwen3` + тот же kwarg).
- `compose.sglang.yml` не проверен на живом GPU-хосте (путь к `qwen3_reranker.jinja` внутри образа, `--reasoning-parser qwen3` для Qwen3.8, размер `--context-length`).

## Корпус
- Легенда плейсхолдеров `{{...}}` в `speech-modules.md` лежит в преамбуле и становится отдельным чанком; вариант — префиксовать к чанкам файла.
- Purge-before-insert идёт по `document_name` (имя файла), не по `file_id`: два файла с одним именем перетирают чанки друг друга.
- Гейт B: сверка мест «на слух» против видео (`oreo-data/GATE-B-REVIEW.md`).
