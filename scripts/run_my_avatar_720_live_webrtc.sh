#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ -n "${ICE_SERVER:-}" ]]; then
  EXTRA_ARGS+=(--ice-server "${ICE_SERVER}")
fi
if [[ -n "${ICE_SERVER_2:-}" ]]; then
  EXTRA_ARGS+=(--ice-server "${ICE_SERVER_2}")
fi
if [[ -n "${ICE_SERVER_3:-}" ]]; then
  EXTRA_ARGS+=(--ice-server "${ICE_SERVER_3}")
fi
if [[ -n "${ICE_TRANSPORT_POLICY:-}" ]]; then
  EXTRA_ARGS+=(--ice-transport-policy "${ICE_TRANSPORT_POLICY}")
fi
if [[ -n "${ICE_USERNAME:-}" ]]; then
  EXTRA_ARGS+=(--ice-username "${ICE_USERNAME}")
fi
if [[ -n "${ICE_CREDENTIAL:-}" ]]; then
  EXTRA_ARGS+=(--ice-credential "${ICE_CREDENTIAL}")
fi

python -m scripts.personaplex_musetalk_webrtc \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8780}" \
  --avatar-id my_avatar_720_live \
  --version v15 \
  --gpu-id "${GPU_ID:-0}" \
  --use-fp16 \
  --require-mmpose \
  --personaplex-host "${PERSONAPLEX_HOST:-127.0.0.1}" \
  --personaplex-port "${PERSONAPLEX_PORT:-8998}" \
  --personaplex-path "${PERSONAPLEX_PATH:-/api/avatar/audio}" \
  --fps "${FPS:-25}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --window-ms "${WINDOW_MS:-640}" \
  --hop-ms "${HOP_MS:-80}" \
  --min-window-ms "${MIN_WINDOW_MS:-320}" \
  --max-advance-ms "${MAX_ADVANCE_MS:-240}" \
  --max-tail-frames "${MAX_TAIL_FRAMES:-5}" \
  --mouth-smoothing-alpha "${MOUTH_SMOOTHING_ALPHA:-0.75}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
