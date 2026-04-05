#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m scripts.realtime_inference \
  --version v15 \
  --gpu_id "${GPU_ID:-0}" \
  --use_fp16 \
  --require_mmpose \
  --non_interactive \
  --inference_config configs/inference/runtime/my_avatar_720_live_step1.yaml \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config models/musetalkV15/musetalk.json \
  --fps "${FPS:-25}" \
  --batch_size "${BATCH_SIZE:-8}" \
  "$@"
