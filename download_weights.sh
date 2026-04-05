#!/bin/bash

set -euo pipefail

# Set the checkpoints directory
CheckpointsDir="models"

# Optional prepared-avatar restore (enabled by default).
# Set RESULTS_RESTORE=0 to skip.
RESULTS_RESTORE="${RESULTS_RESTORE:-1}"
RESULTS_ARCHIVE_URL="${RESULTS_ARCHIVE_URL:-https://drive.google.com/open?id=12aWbN28KSYCb3c9YJQxxLLi3I-6PSAnv}"
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
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# Download MuseTalk V1.5 weights (unet.pth)
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# Download SD VAE weights
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
huggingface-cli download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download DWPose weights
huggingface-cli download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"

# Download SyncNet weights
huggingface-cli download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt"

# Download Face Parse Bisent weights
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

echo "All weights have been downloaded successfully."

if [ "$RESULTS_RESTORE" = "1" ]; then
  mkdir -p "$RESULTS_EXTRACT_DIR"
  tmp_archive="$(mktemp /tmp/${RESULTS_AVATAR_ID}.XXXXXX.tar)"

  echo "Downloading prepared avatar archive for '${RESULTS_AVATAR_ID}'..."
  gdown --fuzzy "$RESULTS_ARCHIVE_URL" -O "$tmp_archive"

  echo "Extracting archive to: $RESULTS_EXTRACT_DIR"
  tar -xf "$tmp_archive" -C "$RESULTS_EXTRACT_DIR"
  rm -f "$tmp_archive"

  if [ ! -f "$RESULTS_EXTRACT_DIR/$RESULTS_AVATAR_ID/avator_info.json" ]; then
    echo "Restore failed: expected file not found -> $RESULTS_EXTRACT_DIR/$RESULTS_AVATAR_ID/avator_info.json"
    exit 1
  fi

  echo "Prepared avatar restored: $RESULTS_EXTRACT_DIR/$RESULTS_AVATAR_ID"
else
  echo "Skipping prepared avatar restore (RESULTS_RESTORE=0)."
fi
