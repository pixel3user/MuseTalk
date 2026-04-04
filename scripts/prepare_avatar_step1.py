import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def build_config_dict(avatar_id: str, video_path: Path, audio_path: Path, bbox_shift: int):
    return {
        avatar_id: {
            "preparation": True,
            "bbox_shift": bbox_shift,
            "video_path": str(video_path),
            "audio_clips": {
                "audio_0": str(audio_path),
            },
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step-1 Avatar preparation runner for MuseTalk realtime pipeline."
    )
    parser.add_argument("--avatar_id", type=str, required=True, help="Unique avatar id")
    parser.add_argument("--video_path", type=str, required=True, help="Input avatar video path")
    parser.add_argument("--audio_path", type=str, required=True, help="Input .wav path for validation run")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift")
    parser.add_argument("--fps", type=int, default=25, help="Output FPS")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument(
        "--version", type=str, default="v15", choices=["v15", "v1"], help="MuseTalk version"
    )
    parser.add_argument(
        "--result_dir", type=str, default="results/realtime", help="Base output directory"
    )
    parser.add_argument(
        "--ffmpeg_path",
        type=str,
        default="tools/ffmpeg",
        help="Directory containing ffmpeg executable",
    )
    parser.add_argument(
        "--skip_save_images",
        action="store_true",
        help="Skip writing intermediate images",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use fp16 on GPU for speed. Default uses fp32 for quality.",
    )
    parser.add_argument(
        "--parsing_mode",
        type=str,
        default="jaw",
        help="Face parsing mode for blending. 'jaw' is recommended for v1.5 quality.",
    )
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face crop in v1.5.")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Left cheek width for face parsing.")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Right cheek width for face parsing.")
    parser.add_argument(
        "--require_mmpose",
        action="store_true",
        help="Fail if mmpose/DWPose is unavailable (recommended for quality-critical runs).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    video_path = Path(args.video_path).expanduser().resolve()
    audio_path = Path(args.audio_path).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio path not found: {audio_path}")

    if audio_path.suffix.lower() != ".wav":
        print(
            f"[WARN] Expected a .wav file for step-1 validation, got: {audio_path.suffix}",
            file=sys.stderr,
        )

    runtime_cfg_dir = repo_root / "configs" / "inference" / "runtime"
    runtime_cfg_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_cfg_dir / f"{args.avatar_id}_step1.yaml"
    config_data = build_config_dict(args.avatar_id, video_path, audio_path, args.bbox_shift)

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, sort_keys=False)

    if args.version == "v15":
        unet_model_path = "models/musetalkV15/unet.pth"
        unet_config = "models/musetalkV15/musetalk.json"
    else:
        unet_model_path = "models/musetalk/pytorch_model.bin"
        unet_config = "models/musetalk/musetalk.json"

    cmd = [
        sys.executable,
        "-m",
        "scripts.realtime_inference",
        "--inference_config",
        str(config_path.relative_to(repo_root)),
        "--result_dir",
        args.result_dir,
        "--unet_model_path",
        unet_model_path,
        "--unet_config",
        unet_config,
        "--version",
        args.version,
        "--fps",
        str(args.fps),
        "--gpu_id",
        str(args.gpu_id),
        "--batch_size",
        str(args.batch_size),
        "--ffmpeg_path",
        args.ffmpeg_path,
        "--non_interactive",
        "--force_recreate_avatar",
        "--parsing_mode",
        args.parsing_mode,
        "--extra_margin",
        str(args.extra_margin),
        "--left_cheek_width",
        str(args.left_cheek_width),
        "--right_cheek_width",
        str(args.right_cheek_width),
    ]
    if args.skip_save_images:
        cmd.append("--skip_save_images")
    if args.use_fp16:
        cmd.append("--use_fp16")
    if args.require_mmpose:
        cmd.append("--require_mmpose")

    print(f"[INFO] Wrote runtime config: {config_path}")
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
