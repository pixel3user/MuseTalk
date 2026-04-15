#!/bin/bash

set -euo pipefail

# Set the checkpoints directory
CheckpointsDir="models"

# Optional prepared-avatar restore (enabled by default).
# Set RESULTS_RESTORE=0 to skip.
RESULTS_RESTORE="${RESULTS_RESTORE:-1}"
RESULTS_ARCHIVE_URL="${RESULTS_ARCHIVE_URL:-https://firebasestorage.googleapis.com/v0/b/farm2market-99.firebasestorage.app/o/results%2Fresults_20260415_114349.tar.gz?alt=media}"
RESULTS_EXTRACT_DIR="${RESULTS_EXTRACT_DIR:-results/v15/avatars}"
RESULTS_AVATAR_ID="${RESULTS_AVATAR_ID:-my_avatar_720_live}"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# Install required packages
pip install -U "huggingface_hub[cli]"
pip install gdown

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.0 weights
hf download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --repo-type model \
  --include "musetalk/musetalk.json" \
  --include "musetalk/pytorch_model.bin"

# Download MuseTalk V1.5 weights (unet.pth)
hf download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --repo-type model \
  --include "musetalkV15/musetalk.json" \
  --include "musetalkV15/unet.pth"

# Download SD VAE weights
hf download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --repo-type model \
  --include "config.json" \
  --include "diffusion_pytorch_model.bin"

# Download Whisper weights
hf download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --repo-type model \
  --include "config.json" \
  --include "pytorch_model.bin" \
  --include "preprocessor_config.json"

# Download DWPose weights
hf download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --repo-type model \
  --include "dw-ll_ucoco_384.pth"

# Download SyncNet weights
hf download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --repo-type model \
  --include "latentsync_syncnet.pt"

# Download Face Parse Bisent weights
gdown https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

required_files=(
  "$CheckpointsDir/musetalk/musetalk.json"
  "$CheckpointsDir/musetalk/pytorch_model.bin"
  "$CheckpointsDir/musetalkV15/musetalk.json"
  "$CheckpointsDir/musetalkV15/unet.pth"
  "$CheckpointsDir/sd-vae/config.json"
  "$CheckpointsDir/sd-vae/diffusion_pytorch_model.bin"
  "$CheckpointsDir/whisper/config.json"
  "$CheckpointsDir/whisper/pytorch_model.bin"
  "$CheckpointsDir/whisper/preprocessor_config.json"
)
for f in "${required_files[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Missing required file after download: $f"
    exit 1
  fi
done

echo "All weights have been downloaded successfully."

if [ "$RESULTS_RESTORE" = "1" ]; then
  mkdir -p "$RESULTS_EXTRACT_DIR"
  tmp_archive="$(mktemp /tmp/${RESULTS_AVATAR_ID}.XXXXXX.tar)"

  echo "Downloading prepared avatar archive for '${RESULTS_AVATAR_ID}'..."
  curl -L "$RESULTS_ARCHIVE_URL" -o "$tmp_archive"

  echo "Extracting archive to: $RESULTS_EXTRACT_DIR"
  tar -xf "$tmp_archive" -C "$RESULTS_EXTRACT_DIR" --strip-components=3
  rm -f "$tmp_archive"

  if [ ! -f "$RESULTS_EXTRACT_DIR/$RESULTS_AVATAR_ID/avator_info.json" ]; then
    echo "Restore failed: expected file not found -> $RESULTS_EXTRACT_DIR/$RESULTS_AVATAR_ID/avator_info.json"
    exit 1
  fi

  echo "Prepared avatar restored: $RESULTS_EXTRACT_DIR/$RESULTS_AVATAR_ID"
else
  echo "Skipping prepared avatar restore (RESULTS_RESTORE=0)."
fi
