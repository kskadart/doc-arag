FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/docarag ./src/docarag

FROM python:3.13-slim-trixie

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src/docarag ./src/docarag

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8103

ENTRYPOINT []
CMD ["uvicorn", "src.docarag.main:app", \
"--host", "0.0.0.0", \
"--port", "8103", \
"--reload", \
"--reload-dir", "/app/src/docarag"]