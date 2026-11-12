#!/usr/bin/env bash
# Load-and-answer smoke test for a gufo build containing this extension.
#
#   tools/load-smoke.sh <gufo> <model-snapshot-dir> [port]
#
# <model-snapshot-dir> is an unsloth/Qwen3.8-Flash-Next-GGUF snapshot with
# UD-IQ4_XS/, MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf and mmproj-BF16.gguf (see
# README.md). The server needs ~81 GiB of device memory: stop other model
# servers first. It serves two sessions at a 262,144-token context, asks three
# questions (the second answer must contain 391) and stops the server.
set -euo pipefail

BIN=${1:?usage: load-smoke.sh <gufo> <model-snapshot-dir> [port]}
MODELS=${2:?usage: load-smoke.sh <gufo> <model-snapshot-dir> [port]}
PORT=${3:-18180}
SLOG=$(mktemp -t flash-next-iq-smoke.XXXXXX.log)
MODEL=$(ls "$MODELS"/UD-IQ4_XS/*-00001-of-*.gguf | head -1)

log() { echo "[load-smoke] $(date +%T) $*"; }
SRV=""
cleanup() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null; wait 2>/dev/null || true; }
trap cleanup EXIT

log "serving $MODEL on 127.0.0.1:$PORT (log: $SLOG)"
"$BIN" serve --host 127.0.0.1 --port "$PORT" llm \
  --model "$MODEL" \
  --speculative mtp --mtp-model "$MODELS/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf" \
  --mmproj "$MODELS/mmproj-BF16.gguf" \
  --sessions 2 --context 262144 \
  --served-model-name Qwen3.8-Flash-Next > "$SLOG" 2>&1 &
SRV=$!

ready=0
for _ in $(seq 1 120); do
  kill -0 "$SRV" 2>/dev/null || { log "server exited:"; tail -25 "$SLOG"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
done
[ "$ready" = 1 ] || { log "timeout"; tail -25 "$SLOG"; exit 1; }
grep -m1 'load_completed' "$SLOG" || true

chat() {
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"Qwen3.8-Flash-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2}"
}

log "chat 1: short answer"
chat "What is the capital of France? Answer in one word." 64; echo
log "chat 2: arithmetic (the prompt must be read correctly)"
answer=$(chat "What is 17*23? Show only the number." 400)
echo "$answer"
grep -q 391 <<<"$answer" || { log "FAIL: no 391 in the answer"; exit 1; }
log "chat 3: MTP-heavy generation"
chat "List three prime numbers and explain why each is prime. Be concise." 160; echo

log "server log (load, MTP acceptance, cache):"
grep -E 'load_completed|event=completed' "$SLOG" | tail -4
log "PASS"
