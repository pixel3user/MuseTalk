#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[INFO] Installing strictly pinned PyTorch..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

echo "[INFO] Installing MMLab stack for high-quality landmark extraction..."
pip install --no-cache-dir -U openmim

echo "[INFO] Installing exact mmcv binary for PyTorch 2.1.0 + CUDA 11.8..."
mim install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

echo "[INFO] Installing the rest of the MMLab stack..."
mim install mmengine
mim install "mmdet==3.3.0"
mim install "mmpose==1.3.2"

echo "[INFO] Downloading DWPose checkpoint..."
mkdir -p models/dwpose
huggingface-cli download yzd-v/DWPose \
  --local-dir models/dwpose \
  --include "dw-ll_ucoco_384.pth"

echo "[OK] mmpose + DWPose setup complete."