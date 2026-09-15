#!/usr/bin/env bash
# Run the three local model servers natively on macOS (Metal) with llama.cpp:
#   chat       Qwen3.8-27B          -> http://localhost:8081/v1
#   embeddings Qwen3-Embedding-8B   -> http://localhost:8082/v1
#   rerank     Qwen3-Reranker-8B    -> http://localhost:8083/v1
# Same HTTP contracts as OpenRouter and SGLang, so doc-arag only needs
# compose.models.yml (api -> host.docker.internal:808x).
#
#   scripts/local_models.sh start [chat|embed|rerank ...]   # default: all three
#   scripts/local_models.sh stop  [chat|embed|rerank ...]
#   scripts/local_models.sh status
#   scripts/local_models.sh logs <chat|embed|rerank>
#
# Requires `brew install llama.cpp`. Weights are pulled from Hugging Face on
# first start into ~/.cache/huggingface/hub (~16 GB + 9 GB + 9 GB).
# Override repos/quants with LLM_HF_REPO, EMBEDDING_HF_REPO, RERANKER_HF_REPO.
set -euo pipefail

LLM_HF_REPO="${LLM_HF_REPO:-unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M}"
EMBEDDING_HF_REPO="${EMBEDDING_HF_REPO:-Qwen/Qwen3-Embedding-8B-GGUF:Q8_0}"
RERANKER_HF_REPO="${RERANKER_HF_REPO:-Voodisss/Qwen3-Reranker-8B-GGUF-llama_cpp:Q8_0}"
LLM_CTX_SIZE="${LLM_CTX_SIZE:-16384}"
EMBEDDING_CTX_SIZE="${EMBEDDING_CTX_SIZE:-8192}"
RERANKER_CTX_SIZE="${RERANKER_CTX_SIZE:-8192}"
RUN_DIR="${RUN_DIR:-$HOME/.local/state/doc-arag-models}"
mkdir -p "$RUN_DIR"

port_of() {
  case "$1" in
    chat) echo 8081 ;;
    embed) echo 8082 ;;
    rerank) echo 8083 ;;
    *) echo "unknown service: $1" >&2; exit 64 ;;
  esac
}

args_for() {
  case "$1" in
    chat)   echo "--hf-repo $LLM_HF_REPO --alias qwen3.8-27b --ctx-size $LLM_CTX_SIZE --jinja --reasoning off --chat-template-kwargs {\"enable_thinking\":false} --n-gpu-layers 999" ;;
    embed)  echo "--hf-repo $EMBEDDING_HF_REPO --alias qwen3-embedding-8b --embeddings --pooling last --ctx-size $EMBEDDING_CTX_SIZE --batch-size 8192 --ubatch-size 8192 --n-gpu-layers 999" ;;
    rerank) echo "--hf-repo $RERANKER_HF_REPO --alias qwen3-reranker-8b --reranking --ctx-size $RERANKER_CTX_SIZE --n-gpu-layers 999" ;;
    *) echo "unknown service: $1" >&2; exit 64 ;;
  esac
}

pid_of() { [ -f "$RUN_DIR/$1.pid" ] && cat "$RUN_DIR/$1.pid" || true; }

is_running() {
  local pid; pid=$(pid_of "$1")
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_one() {
  local name=$1
  if is_running "$name"; then echo "$name already running (pid $(pid_of "$name"), port $(port_of "$name"))"; return; fi
  command -v llama-server >/dev/null || { echo "llama-server not found: brew install llama.cpp" >&2; exit 1; }
  # shellcheck disable=SC2086
  nohup llama-server $(args_for "$name") --host 127.0.0.1 --port "$(port_of "$name")" \
    > "$RUN_DIR/$name.log" 2>&1 &
  echo $! > "$RUN_DIR/$name.pid"
  echo "$name starting on :$(port_of "$name") (pid $!, log $RUN_DIR/$name.log)"
}

stop_one() {
  local name=$1 pid; pid=$(pid_of "$name")
  if [ -n "$pid" ] && kill "$pid" 2>/dev/null; then echo "$name stopped (pid $pid)"; else echo "$name not running"; fi
  rm -f "$RUN_DIR/$name.pid"
}

status_one() {
  local name=$1 state="down" health=""
  if is_running "$name"; then
    state="pid $(pid_of "$name")"
    health=$(curl -s -m 2 "http://127.0.0.1:$(port_of "$name")/health" || echo "(loading)")
  fi
  printf '%-7s :%s  %s  %s\n' "$name" "$(port_of "$name")" "$state" "$health"
}

cmd=${1:-status}; shift || true
services=("$@"); [ ${#services[@]} -eq 0 ] && services=(chat embed rerank)

case "$cmd" in
  start)  for s in "${services[@]}"; do start_one "$s"; done ;;
  stop)   for s in "${services[@]}"; do stop_one "$s"; done ;;
  status) for s in chat embed rerank; do status_one "$s"; done ;;
  logs)   tail -n 50 -f "$RUN_DIR/${services[0]}.log" ;;
  *) sed -n '2,14p' "$0"; exit 64 ;;
esac
