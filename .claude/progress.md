# Прогресс (обновлять после каждого этапа)

Ветка: `feat/api-model-providers` (от `feat/md-domain-support` = PR #12). План: `~/.claude/plans/rag-bright-seal.md`.

| Этап | Статус | Дата | Примечание |
|---|---|---|---|
| 0. Закрыть старое (merge PR #12, #6; baseline) | частично | 2026-09-13 | baseline на ветке: 99 passed / 4 skipped, ruff+mypy чистые. **Merge PR #12 и #6 — ждёт команды пользователя.** |
| 1. CLAUDE.md + .claude/ | done | 2026-09-13 | |
| 2. Settings + LLM-фабрика (langchain-openai) | done | 2026-09-13 | `langchain-openai==1.6.2`, langchain-core 1.6.0→1.6.3 |
| 3. Эмбеддинги по HTTP, удаление gRPC-эмбеддинга | done | 2026-09-13 | `clients/embedding_http.py`, gRPC-эмбеддинг и его pb2 удалены |
| 4. Реранкер: grpc / openai-rerank / none | done | 2026-09-13 | `clients/reranker_http.py`, `RerankerError` + fallback |
| 5. Weaviate lifecycle + фиксы retrieval + QueryRequest.domain | done | 2026-09-13 | insert_many, fail-loud, recreate, verify dimension, target_vector, domain-фильтр |
| 6. Чанкинг 1800/200 + scripts/load_corpus.py + run_control_questions.py | done | 2026-09-13 | dry-run: 14 файлов / 9 доменов / 154 чанка / 0 над порогом; отчёт `oreo-data/docs/chunking-report.md` |
| 6b. Дефолты под выбор моделей (OpenRouter Qwen ×3, `OPENROUTER_API_KEY`) | done | 2026-09-13 | контракты rerank у OpenRouter и llama.cpp совпали с клиентом |
| 6c. Локальные модели: `scripts/local_models.sh` (llama.cpp на Metal) + `compose.models.yml`; `compose.sglang.yml` для GPU-сервера; клиент rerank понимает диалект SGLang | done | 2026-09-13 | llama.cpp 0.4.0 через brew, все три проверены на Metal: эмбеддер dim 4096, 9 чанков за 0.7 с; реранкер (GGUF `Voodisss/...-llama_cpp` рабочий) релевантные 0.999/0.986, нерелевантные ~0, 0.4 с на 4 кандидата; чат Qwen3.8-27B UD-Q4_K_M 26 ток/с, thinking выключен флагами `--reasoning off --chat-template-kwargs` |
| 6a. Golden set (`oreo-data/golden/`) + `scripts/eval_golden.py` + `sources` в `/query` | done | 2026-09-13 | 85 записей / 9 доменов (factual 23, paraphrase 26, procedural 18, cross-doc 9, negative 9), все проверены против корпуса; `golden-set.jsonl` собран; прогон ждёт этап 7 |
| 7. Загрузка и верификация (пилот → корпус → 12 вопросов) | blocked | | ждёт: ключ OpenRouter, локальный эмбеддер (Ollama), Docker Desktop; dry-run уже пройден |

Состояние проверок: `make tests linter typecheck` — 139 passed / 4 skipped, ruff и mypy чистые. **Закоммичено 2026-09-13** на `feat/api-model-providers`: `e92dd81` providers, `02b4208` scripts, `443bfcd` docs/compose; oreo-data `38204af` golden set. Не запушено.

## Блокеры на пользователе
- `OPENROUTER_API_KEY` — один ключ на chat + embeddings + rerank (`qwen/qwen3.8-flash`, `qwen/qwen3-embedding-8b`, `qwen/qwen3-reranker-8b`; всё проверено в каталоге OpenRouter 2026-09-13).
- Docker Desktop не запущен (нужен для weaviate/minio/api). Локальные модели — нативно, см. 6c.
- Команды на merge PR #12 (doc-arag) и PR #6 (rag-services).

## Среда (2026-09-13)
arm64 Mac, 128 GB. Установлено в этой сессии: `uv 0.12.13` (brew), `libmagic 5.48` (brew).
