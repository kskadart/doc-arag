# Backlog (вне текущего скоупа; из аудита 2026-09-13)

## Агент
- ~~Итерации бесполезны: второй круг повторяет первый~~ — закрыто 2026-09-14 (ветка `feat/chat-sessions`): rephrase получает `previous_queries` и просит другую формулировку.
- Промпт evaluate — inline f-string без system message; condense/generate/summary уже константы в `agent.py` / `sessions.py`, вынести всё в один модуль шаблонов.
- Чат-сессии без владельца: `/sessions/{id}` под `user_router` (решение №12), но любой вошедший пользователь, знающий id, читает и удаляет чужую историю, а `/query` с чужим `session_id` дописывает в неё. Нужен `owner` (`Remote-User`) у сообщений в `ChatMessages` и проверка в store; клиент генерирует uuid, где доступен `crypto.randomUUID`, что снижает риск угадывания.
- TTL-sweep сессий идёт в каждом uvicorn-воркере (`--workers 2` в prod) — идемпотентно, но лишние вызовы; при оживлении прода вынести в один воркер или cron.
- Weaviate 1.39 умеет native `object_ttl_config`; когда фича стабилизируется, заменить ручной `delete_many` по `updated_at`.

## Эксплуатация
- In-memory task store (`task_progress.py`) + `--workers 2` в `docker/Dockerfile.prod` → `GET /tasks/{id}` 404 на «чужом» воркере; нет вытеснения записей.
- Weaviate `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true` и MinIO дефолтный пароль в `docker/compose.prod.yml` (порты только на 127.0.0.1, API защищён секретом прокси, но Weaviate и MinIO внутри общей docker-сети открыты любому соседнему контейнеру — вынести api/weaviate/minio в отдельную сеть, в общей оставить только api); rate-limit на API нет (Authelia даёт только брутфорс-защиту логина).
- FastAPI парсит multipart-тело до зависимостей, поэтому `require_admin` на `/uploads` срабатывает после приёма файла; не-админов отсекает раньше Authelia на границе, но при прямом доступе в сеть большой файл всё равно будет принят до 403.
- Чат-сессии не привязаны к пользователю: `DELETE /sessions/{id}` и память чата доступны любому залогиненному, знающему uuid. Добавить `owner` из `CurrentUser` в `feat/chat-sessions`.
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

## Качество (после v5, 2026-09-15)
- cross-doc 4.00: diag-010 не находит ticket-systems.md (второй документ не попадает в топ даже при поиске по двум векторам); sla-003 отвечает слишком коротко — новый промпт местами урезал полноту (16 ответов short).
- diag-006: реранкер даёт всем кандидатам ~0.04 из-за отвлекающих слов про цену и сотрудника.
- Поиск по двум векторам не доказал пользы — абляция или выключить по умолчанию (лишний вызов эмбеддера на OpenRouter).
- `fact_coverage` в `scripts/eval_golden.py` даёт ложные промахи на markdown-выделении и словоформах — снимать разметку перед сравнением.
- Качество и задержка на моделях OpenRouter не измерены — прогон golden до деплоя.

## Деплой (Selectel, CPU)
- **Эмбеддинги OpenRouter маршрутизируются разным провайдерам** (решено 2026-09-16: `EMBEDDING_EXTRA_BODY` закрепляет Nebius → DeepInfra, замер 0.4–1.7 с): Nebius отвечает за 0.7–1.5 с, DeepInfra — до 25 с; параллельный второй вектор (поиск по оригиналу и перефразу) чаще попадает на медленного, запрос тогда идёт 40 с. Варианты: закрепить провайдера (`provider.order`/`sort` в теле запроса эмбеддингов — клиенту нужен `EMBEDDING_EXTRA_BODY`), выключить `RETRIEVAL_USE_ORIGINAL_QUERY` (пользы не доказал).
- **flash на negative-вопросах хуже локальной 27B** (Opus 4.00 против 4.89): ответ одной фразой «в базе нет» без того, что сказать клиенту, либо советы не из корпуса. Кандидат на правку промпта под OpenRouter.
- Переменная `EMBEDDING_SERVICE_URL` в окружении production doc-arag устарела; `ANTHROPIC_*` секреты не нужны при OpenRouter.
- **Qwen3.8-flash на OpenRouter по умолчанию размышляет**: генерация 152 с вместо ~4 с, в 5 раз дороже. На сервере обязательно `LLM_EXTRA_BODY={"reasoning": {"enabled": false}}` (проверено 2026-09-15: пробный вопрос 180 с → 12 с). Поиск по двум векторам на OpenRouter стоит ~2–3 с на лишний вызов эмбеддера.
- Дефолтный OpenRouter-путь `compose.yml` не стартовал: `LLM_EXTRA_BODY=${LLM_EXTRA_BODY:-}` давал пустую строку, pydantic требовал dict. Исправлено 2026-09-15 валидатором в `settings.py`; при деплое проверить, что серверный compose не передаёт другие пустые JSON-переменные.
- Блок `sterility.ai.kskada.com` (второй проект пользователя, Streamlit) убран из Caddyfile, compose и deploy клиента в ветке `feat/selectel-deploy` — на Selectel переезжает только rag. Переменные `STREAMLIT_*` в окружении production клиента остались, не используются.
- Серверный compose не публикует порты наружу (сделано 2026-09-15): api и Weaviate на 127.0.0.1, MinIO без портов. Наружу только Caddy 80/443.
- Облачный файрвол Selectel не фильтрует трафик на прямой публичный IP — нужны группы безопасности или файрвол ОС.

## Интерфейс (после сквозной проверки 2026-09-16)
- Чат в чистом браузере не показывает поле ввода, пока не нажать «New Chat», хотя приветствие просит задать вопрос (`app/[locale]/chat/page.tsx`: `ChatInput` только при `currentSession`).
- Страница документов выводит у каждой карточки сырые заголовки MinIO/S3 (`Accept-Ranges`, `X-Amz-*`, `Strict-Transport-Security`) — шум для администратора.
- Оператору показываются служебные «Rephrased Query» и «Confidence» — решить, нужно ли это в проде.
