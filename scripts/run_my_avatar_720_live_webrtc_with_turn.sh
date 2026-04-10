#!/usr/bin/env bash
set -euo pipefail

TURN_ENV="/opt/onebox/turn-native/turn_credentials.env"
if [[ ! -f "$TURN_ENV" ]]; then
  echo "TURN credentials file not found: $TURN_ENV" >&2
  exit 1
fi

source "$TURN_ENV"

if [[ -z "${TURN_HOST:-}" || -z "${TURN_USER:-}" || -z "${TURN_PASS:-}" ]]; then
  echo "TURN credentials are incomplete in $TURN_ENV" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

ICE_SERVER="turn:${TURN_HOST}:${TURN_PORT:-3478}?transport=tcp" \
ICE_USERNAME="$TURN_USER" \
ICE_CREDENTIAL="$TURN_PASS" \
ICE_TRANSPORT_POLICY=relay \
./scripts/run_my_avatar_720_live_webrtc.sh "$@"
