#!/bin/bash

echo "entrypoint.sh"
whoami
which python
source /opt/conda/etc/profile.d/conda.sh
conda activate musev
which python

if [[ "${RUNTIME_MODE:-gradio}" == "webrtc_api" ]]; then
  if [[ -z "${AVATAR_ID:-}" ]]; then
    echo "AVATAR_ID is required when RUNTIME_MODE=webrtc_api" >&2
    exit 1
  fi
  exec python -m scripts.personaplex_musetalk_webrtc \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8780}" \
    --avatar-id "${AVATAR_ID}" \
    --version "${VERSION:-v15}" \
    --gpu-id "${GPU_ID:-0}" \
    --input-source "${INPUT_SOURCE:-webrtc}" \
    "${@}"
fi

exec python app.py
