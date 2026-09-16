# TODO List

## Weaviate Connection Timeout Fix

### [x] Fixed - Disable PyPI version check in Weaviate client
- **Issue**: Weaviate client was timing out during startup trying to verify package version on PyPI
- **Solution**: Added `skip_init_checks=True` to `weaviate.use_async_with_local()` in `src/docarag/clients/vector_db_client.py`
- **Status**: ✅ Completed
- **File**: `src/docarag/clients/vector_db_client.py`

### [ ] Test application startup with Docker container
- Verify the application starts without `httpx.ConnectTimeout` errors
- Ensure Weaviate container is properly initialized
- **Status**: Pending

### [ ] Verify Weaviate is ready and accessible after startup
- Check that `client.is_ready()` returns True
- Verify connection pooling works correctly
- **Status**: Pending

### [ ] Monitor logs for any connection warnings or resource leaks
- Watch for ResourceWarning messages related to TCP connections
- Verify proper cleanup of Weaviate connections
- **Status**: Pending

## LangGraph RAG Agent

### [x] Implement multi-step RAG agent with LangGraph
- **Solution**: Created `src/docarag/services/agent.py` with stateless LangGraph workflow
- **Features**:
  - Query rephrasing using Claude for optimization
  - Embedding generation via gRPC service
  - Vector similarity search in Weaviate
  - Answer generation with context
  - Quality evaluation and iterative improvement
- **Status**: ✅ Completed
- **File**: `src/docarag/services/agent.py`

### [x] Update /query API endpoint
- **Solution**: Implemented endpoint to use the LangGraph agent
- **Status**: ✅ Completed
- **File**: `src/docarag/api.py`

### [x] Add agent configuration settings
- **Solution**: Added agent-specific settings (confidence threshold, temperature, reranker URL)
- **Status**: ✅ Completed
- **File**: `src/docarag/settings.py`

### [ ] Implement reranker client for external reranking service
- **Goal**: Create `src/docarag/clients/reranker_client.py` similar to EmbeddingGRPCClient
- **Requirements**:
  - Accept query text and list of candidate documents
  - Return top_k reranked results with scores
  - Use `rerank_top_k` from settings
  - Support both sync and async operations
- **Integration**: Add rerank node between retrieve and generate in agent graph
- **Status**: Pending (to be implemented later)

## Доработки после запуска прода (2026-09-16)

Прод: https://oreo.kskada.com — Selectel, модели OpenRouter через внешний прокси, корпус 92 чанка.
Смоук из golden на проде: 15 вопросов, 15/15 нашли нужный документ, факты 89%, утечек нет, медиана 16 с.

### Надёжность
- [ ] Мониторинг: внешний health-чек, алерт на падение прокси и на 403 от OpenRouter.
- [ ] Бэкап томов Weaviate и MinIO.
- [ ] Прокси — единственная точка отказа: нужен запасной маршрут или второй адрес.
- [ ] Сменить пароль прокси и обновить секрет `OUTBOUND_HTTPS_PROXY` (прозвучал в переписке).

### Качество ответов
- [ ] Полный прогон golden (85 вопросов) против прода; сравнить с локальными 4.74 по Opus.
- [ ] Вопросы на два документа (4.00): `diag-010` не находит `ticket-systems.md`, `sla-003` отвечает слишком коротко.
- [ ] Вопросы-отказы на flash слабее локальной модели (4.00 против 4.89): отвечает «в базе нет» без подсказки оператору.
- [ ] Многословие: часть ответов тянет соседние документы в узкий вопрос.
- [ ] Поиск по двум векторам пользы не доказал — абляция или выключить (лишний вызов эмбеддера).
- [ ] `fact_coverage` в `scripts/eval_golden.py` даёт ложные промахи на markdown и словоформах.

### Эксплуатация
- [ ] Загрузчик корпуса: положить `scripts/` в продовый образ либо ходить по SSH-туннелю (сейчас `docker cp` в контейнер).
- [ ] Убрать из production устаревшее: переменная `EMBEDDING_SERVICE_URL`, секреты `ANTHROPIC_*`, переменные `STREAMLIT_*` в клиенте.
- [ ] Почистить локальные остатки: тома Docker от промежуточных стеков, `authelia/users.yml.bak`.

### Интерфейс
- [ ] Поле ввода в чате появляется только после «New Chat», хотя приветствие просит задать вопрос.
- [ ] На странице документов печатаются сырые заголовки S3 (`X-Amz-*`, `Accept-Ranges`).
- [ ] Решить, нужны ли оператору служебные «Rephrased Query» и «Confidence».

### Безопасность
- [ ] Чат-сессии без владельца: `/sessions/{id}` читает и удаляет любой вошедший, знающий uuid.
- [ ] Purge-before-insert идёт по имени файла: два документа с одним именем перетирают чанки друг друга.

### Данные
- [ ] oreo-data PR #1 (golden set, 85 записей) — влить или закрыть.
- [ ] Гейт B: сверка мест «на слух» с видео (`oreo-data/GATE-B-REVIEW.md`).
