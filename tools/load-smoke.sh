#!/usr/bin/env bash
# Final IQ4_XS load test for the flash_next_iq Gufo extension (native v2).
# The ghcr gufo-runtime image has no glibc loader for this Ubuntu-built
# binary, so we run the binary directly in the host ROCm environment while
# production is briefly stopped. Run as yashi; needs docker CLI only to
# stop/start the production container.
set -u

BIN=/home/yashi/Server/gufo/build/release/gufo
MODELS=/var/lib/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66
PORT=18180
SLOG="$(dirname "$0")/gufo-iq-smoke-serve.log"

log() { echo "[iq-smoke] $(date +%T) $*"; }
restore() {
  [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null
  sleep 3
  log "restarting production gufo"
  docker start gufo && log "production restored"
}
trap restore EXIT

ulimit -l unlimited 2>/dev/null || log "memlock stays at $(ulimit -l) KiB (testing anyway)"

log "stopping production gufo (frees ~78 GiB)"
docker stop gufo

log "starting extension-built binary natively on :$PORT"
LD_LIBRARY_PATH=/opt/rocm/lib \
"$BIN" serve --host 0.0.0.0 --port $PORT llm \
  --model "$MODELS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf" \
  --speculative mtp --mtp-model "$MODELS/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf" \
  --mmproj "$MODELS/mmproj-BF16.gguf" \
  --sessions 2 --context 262144 \
  --served-model-name Qwen3.8-Flash-Next \
  --cache-ram-bytes 12884901888 > "$SLOG" 2>&1 &
SRV=$!

log "waiting for model load (see $SLOG)"
for i in $(seq 1 90); do
  kill -0 $SRV 2>/dev/null || { log "serve died:"; tail -25 "$SLOG"; exit 1; }
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
done
[ "${ready:-}" = 1 ] || { log "timeout"; tail -25 "$SLOG"; exit 1; }
grep -i "load\|provider\|extension" "$SLOG" | head -8

chat() {
  curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"Qwen3.8-Flash-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":${2:-64}}"
}

log "chat 1 (IQ4_XS weights):"
chat "What is the capital of France? Answer in one word." 16
echo; log "chat 2 (MTP-heavy generation):"
chat "List three prime numbers and explain why each is prime. Be concise." 160
echo; log "chat 3 (continuation over shared prefix):"
chat "List three prime numbers and explain why each is prime. Be concise. Then add a fourth." 160
echo

log "server log tail (load time, MTP acceptance, cache):"
tail -40 "$SLOG"
log "ALL DONE — trap will restore production"
