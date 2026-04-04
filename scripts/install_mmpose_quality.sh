#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[INFO] Installing MMLab stack for high-quality landmark extraction..."
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

echo "[INFO] Downloading DWPose checkpoint..."
mkdir -p models/dwpose
huggingface-cli download yzd-v/DWPose \
  --local-dir models/dwpose \
  --include "dw-ll_ucoco_384.pth"

echo "[OK] mmpose + DWPose setup complete."
