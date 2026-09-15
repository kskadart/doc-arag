# Деплой на Selectel (CPU)

Сервер: Selectel, Ubuntu 26.04, 2 vCPU / 4 ГБ / 25 ГБ, прямой публичный IP `79.141.79.95`, SSH-алиас `oreon`.
Модели — только по API OpenRouter (Qwen3.8-flash без размышлений, Qwen3-Embedding-8B, Qwen3-Reranker-8B).

## Что крутится на сервере

| Каталог | Compose | Контейнеры | Наружу |
|---|---|---|---|
| `/opt/doc-arag` | `docker/compose.prod.yml` (этот репозиторий) | api, weaviate, minio | ничего: api, Weaviate и MinIO слушают только `127.0.0.1` |
| `/opt/doc-arag-client` | `compose.yml` (doc-arag-client) | client, caddy | Caddy 80/443 |

Общая сеть `arag-common-network`. Бэкенд выкатывается первым: он создаёт сеть.
Docker публикует порты в обход ufw, поэтому публичные порты есть только у Caddy.

## Однократно на сервере

```bash
ssh oreon
apt update && apt -y upgrade
ufw allow OpenSSH && ufw allow 80,443/tcp && ufw enable
curl -fsSL https://get.docker.com | sh
cat > /etc/docker/daemon.json <<'JSON'
{"log-driver": "json-file", "log-opts": {"max-size": "10m", "max-file": "3"}}
JSON
systemctl restart docker
```

## GitHub, окружение `production`

doc-arag:

| Имя | Тип | Значение |
|---|---|---|
| `SERVER_HOST` | secret | `79.141.79.95` |
| `SERVER_USER` | secret | `root` |
| `SSH_PRIVATE_KEY` | secret | отдельный deploy-ключ CI (не личный ключ) |
| `OPENROUTER_API_KEY` | secret | ключ OpenRouter |
| `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | secret | учётка MinIO |
| `AUTH_PROXY_SECRET` | secret | **обязателен**: в `compose.prod.yml` `AUTH_TRUSTED_HEADERS` по умолчанию `true`, и без секрета api падает на старте. То же значение — в doc-arag-client |
| `AUTH_TRUSTED_HEADERS` | var, необязательно | `true` (прод за Caddy + Authelia); `false` — только если осознанно запускаем без авторизации |
| `LLM_EXTRA_BODY` | var, необязательно | по умолчанию `{"reasoning": {"enabled": false}}` |
| `RERANK_TOP_K` | var, необязательно | по умолчанию `8` |

doc-arag-client: `SERVER_HOST`, `SERVER_USER`, `SSH_PRIVATE_KEY` (те же), vars `DOMAIN=rag.kskada.com`, `BACKEND_API_HOST=api`, `BACKEND_API_PORT=8103`, `API_PATH=/api`.
Авторизация (Authelia): var `AUTH_MODE=on`, secrets `AUTH_PROXY_SECRET` (как у бэкенда), `AUTHELIA_SESSION_SECRET`, `AUTHELIA_STORAGE_ENCRYPTION_KEY`, `AUTHELIA_JWT_SECRET` (каждый `openssl rand -hex 32`), `AUTHELIA_USERS_YML_B64` (base64 от `authelia/users.yml`). Без `AUTH_MODE=on` сайт открыт всем: вопросы за счёт OpenRouter, загрузка и удаление документов.

Переменная `EMBEDDING_SERVICE_URL` в окружении doc-arag устарела и не используется.

## DNS (reg.ru)

A-записи `rag.kskada.com` и `auth.rag.kskada.com` (портал Authelia) → `79.141.79.95`. Пока они указывают на старый сервер или отсутствуют, Caddy не выпустит сертификаты.

## Выкладка

1. Workflow `Deploy` в doc-arag (push в `main` или ручной запуск). Образ тегируется полным SHA коммита, сервер тянет именно его.
2. Workflow `Deploy` в doc-arag-client.
3. Загрузка корпуса через SSH-туннель (api и Weaviate на сервере слушают только loopback):

```bash
ssh -N -L 8103:127.0.0.1:8103 -L 8080:127.0.0.1:8080 oreon &
uv run python -m scripts.load_corpus        # по умолчанию localhost:8103 и localhost:8080
```

4. Проверка: `curl https://rag.kskada.com/api/health`.

## Откат

Перезапустить workflow `Deploy` у предыдущего успешного коммита: образ с его SHA уже лежит в ghcr.io.
