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
if [[ -n "${PERSONAPLEX_TEXT_PROMPT:-}" ]]; then
  EXTRA_ARGS+=(--personaplex-text-prompt "${PERSONAPLEX_TEXT_PROMPT}")
fi
if [[ -n "${PERSONAPLEX_VOICE_PROMPT:-}" ]]; then
  EXTRA_ARGS+=(--personaplex-voice-prompt "${PERSONAPLEX_VOICE_PROMPT}")
fi
if [[ -n "${PERSONAPLEX_EXTRA_QUERY:-}" ]]; then
  EXTRA_ARGS+=(--personaplex-extra-query "${PERSONAPLEX_EXTRA_QUERY}")
fi
if [[ -n "${API_TOKEN:-}" ]]; then
  EXTRA_ARGS+=(--enable-api-auth --api-token "${API_TOKEN}")
fi
if [[ -n "${SESSION_OFFER_TIMEOUT_SECONDS:-}" ]]; then
  EXTRA_ARGS+=(--session-offer-timeout-seconds "${SESSION_OFFER_TIMEOUT_SECONDS}")
fi
if [[ -n "${SESSION_MAX_AGE_SECONDS:-}" ]]; then
  EXTRA_ARGS+=(--session-max-age-seconds "${SESSION_MAX_AGE_SECONDS}")
fi
if [[ -n "${SESSION_CLEANUP_INTERVAL_SECONDS:-}" ]]; then
  EXTRA_ARGS+=(--session-cleanup-interval-seconds "${SESSION_CLEANUP_INTERVAL_SECONDS}")
fi
if [[ "${MULTI_SESSION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--multi-session)
fi
if [[ "${REQUIRE_MMPOSE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--require-mmpose)
fi
if [[ "${WEB_TEST_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--web-test-only)
fi
if [[ "${DEBUG_WEBRTC:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--debug)
fi
if [[ -n "${DEBUG_EVENTS_LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--debug-events-limit "${DEBUG_EVENTS_LIMIT}")
fi

python -m scripts.personaplex_musetalk_webrtc \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8780}" \
  --avatar-id my_avatar_720_live \
  --version v15 \
  --gpu-id "${GPU_ID:-0}" \
  --use-fp16 \
  --personaplex-host "${PERSONAPLEX_HOST:-127.0.0.1}" \
  --personaplex-port "${PERSONAPLEX_PORT:-8998}" \
  --personaplex-path "${PERSONAPLEX_PATH:-/api/chat}" \
  --input-source "${INPUT_SOURCE:-mirror}" \
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
