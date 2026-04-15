@echo off
setlocal

:: Set the checkpoints directory
set CheckpointsDir=models

:: Create necessary directories
mkdir %CheckpointsDir%\musetalk
mkdir %CheckpointsDir%\musetalkV15
mkdir %CheckpointsDir%\syncnet
mkdir %CheckpointsDir%\dwpose
mkdir %CheckpointsDir%\face-parse-bisent
mkdir %CheckpointsDir%\sd-vae
mkdir %CheckpointsDir%\whisper

:: Install required packages
pip install -U "huggingface_hub[hf_xet]"

:: Set HuggingFace endpoint
set HF_ENDPOINT=https://hf-mirror.com

:: Download MuseTalk weights
hf download TMElyralab/MuseTalk --local-dir %CheckpointsDir% --repo-type model "musetalk/musetalk.json" "musetalk/pytorch_model.bin" "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

:: Download SD VAE weights
hf download stabilityai/sd-vae-ft-mse --local-dir %CheckpointsDir%\sd-vae --repo-type model "config.json" "diffusion_pytorch_model.bin"

:: Download Whisper weights
hf download openai/whisper-tiny --local-dir %CheckpointsDir%\whisper --repo-type model "config.json" "pytorch_model.bin" "preprocessor_config.json"

:: Download DWPose weights
hf download yzd-v/DWPose --local-dir %CheckpointsDir%\dwpose --repo-type model "dw-ll_ucoco_384.pth"

:: Download SyncNet weights
hf download ByteDance/LatentSync --local-dir %CheckpointsDir%\syncnet --repo-type model "latentsync_syncnet.pt"

:: Download face-parse-bisent weights
hf download ManyOtherFunctions/face-parse-bisent --local-dir %CheckpointsDir%\face-parse-bisent --repo-type model "79999_iter.pth" "resnet18-5c106cde.pth"

echo All weights have been downloaded successfully!
endlocal 
