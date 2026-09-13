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
| 7. Загрузка корпуса (локальный стек) | done | 2026-09-14 | Docker Desktop + `compose.models.yml`; MinIO переехал на quay.io (пин `RELEASE.2025-09-07T16-13-09Z`); пилот 9 чанков, весь корпус 154 чанка / 9 доменов за 27 с; `/query` ≈ 31 с (3 LLM-вызова + rerank) |
| 8. Golden baseline v1 (чанки по заголовкам 1800/200, k=20→5, локальные Qwen) | done | 2026-09-14 | **doc_hit 98%, domain_hit 98%, facts 70%, forbidden 2**, ~23 с/запрос. Причины провалов: 10/85 ответов по-английски, многословие «Based on…/Document N», 1 реальная утечка суммы (лимит 20 000 руб из корпуса), 2 промаха retrieval (tarif-010, diag-006 — ответ из соседнего файла). Отчёт `.claude/reports/golden-baseline-v1.md` (gitignored) |
| 8a. Тюнинг v2: русский system-prompt генерации, метки фрагментов по-русски; в eval — валюта считается утечкой только рядом с числом, мягкое покрытие фактов по основам слов | done | 2026-09-14 | полный v2: doc_hit 98%, facts 71% (76% мягко), forbidden 0, **judge 4.20/5 (84%), доля ≥4 — 81%**; 15/16 слабых — «верно, но опущены детали» (следствие правила «сжато») |
| 8b. Тюнинг v3: промпт на полноту (все шаги/условия, при отсутствии цифры — что говорит база), контекст 5→8 фрагментов (`RERANK_TOP_K=8` в compose.models.yml) | in progress | 2026-09-14 | на 16 слабых v2: judge 3.0→4.06, facts 39→66%; полный прогон — `golden-v3-full.md` |
| 9. Дальнейший тюнинг | todo | | кандидаты: запрос-инструкция для Qwen3-Embedding (готово в ветке `feat/retrieval-tuning`, worktree `.claude/worktrees/tuning`, не влито — retrieval и так 98%), альтернативные источники в golden (`alt_sources`), rerank_top_k, judge-калибровка, сравнение с OpenRouter-моделями |

Состояние проверок: `make tests linter typecheck` — 139 passed / 4 skipped, ruff и mypy чистые. **Закоммичено 2026-09-13** на `feat/api-model-providers`: `e92dd81` providers, `02b4208` scripts, `443bfcd` docs/compose; oreo-data `38204af` golden set. Не запушено.

## Блокеры на пользователе
- Ключ OpenRouter — в самом конце, по слову пользователя (не напоминать).
- Команды на merge PR #12 (doc-arag) и PR #6 (rag-services), push.

## Среда (2026-09-13)
arm64 Mac, 128 GB. Установлено в этой сессии: `uv 0.12.13` (brew), `libmagic 5.48` (brew).
