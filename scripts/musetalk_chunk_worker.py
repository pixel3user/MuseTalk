import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import torch

RT = None


def get_rt():
    global RT
    if RT is None:
        import scripts.realtime_inference as rt_module

        RT = rt_module
    return RT


def build_runtime_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        version=cli_args.version,
        ffmpeg_path=cli_args.ffmpeg_path,
        gpu_id=cli_args.gpu_id,
        vae_type=cli_args.vae_type,
        unet_config=cli_args.unet_config,
        unet_model_path=cli_args.unet_model_path,
        whisper_dir=cli_args.whisper_dir,
        inference_config="",
        bbox_shift=cli_args.bbox_shift,
        result_dir=cli_args.result_dir,
        extra_margin=cli_args.extra_margin,
        fps=cli_args.fps,
        audio_padding_length_left=cli_args.audio_padding_length_left,
        audio_padding_length_right=cli_args.audio_padding_length_right,
        batch_size=cli_args.batch_size,
        output_vid_name=None,
        use_saved_coord=False,
        saved_coord=False,
        parsing_mode=cli_args.parsing_mode,
        left_cheek_width=cli_args.left_cheek_width,
        right_cheek_width=cli_args.right_cheek_width,
        skip_save_images=cli_args.skip_save_images,
        non_interactive=True,
        force_recreate_avatar=False,
        use_fp16=cli_args.use_fp16,
        require_mmpose=cli_args.require_mmpose,
    )


def ensure_ffmpeg(args: argparse.Namespace) -> None:
    rt = get_rt()
    if rt.fast_check_ffmpeg():
        return
    path_separator = ";" if os.name == "nt" else ":"
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ.get('PATH', '')}"
    if not rt.fast_check_ffmpeg():
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or pass --ffmpeg-path with a valid executable directory."
        )


def setup_runtime(cli_args: argparse.Namespace) -> None:
    rt = get_rt()
    rt.args = build_runtime_args(cli_args)
    ensure_ffmpeg(rt.args)
    rt.device = torch.device(f"cuda:{cli_args.gpu_id}" if torch.cuda.is_available() else "cpu")

    if rt.args.require_mmpose and not rt.MMPOSE_AVAILABLE:
        raise RuntimeError("mmpose/DWPose is required but unavailable in this runtime.")
    if not rt.MMPOSE_AVAILABLE:
        print("[worker][warn] mmpose unavailable; using fallback face detector.")

    rt.vae, rt.unet, rt.pe = rt.load_all_model(
        unet_model_path=rt.args.unet_model_path,
        vae_type=rt.args.vae_type,
        unet_config=rt.args.unet_config,
        device=rt.device,
    )
    rt.timesteps = torch.tensor([0], device=rt.device)

    if rt.device.type == "cuda" and rt.args.use_fp16:
        rt.pe = rt.pe.half().to(rt.device)
        rt.vae.vae = rt.vae.vae.half().to(rt.device)
        rt.unet.model = rt.unet.model.half().to(rt.device)
        print("[worker] precision: fp16")
    else:
        rt.pe = rt.pe.float().to(rt.device)
        rt.vae.vae = rt.vae.vae.float().to(rt.device)
        rt.unet.model = rt.unet.model.float().to(rt.device)
        print("[worker] precision: fp32")

    rt.audio_processor = rt.AudioProcessor(feature_extractor_path=rt.args.whisper_dir)
    rt.weight_dtype = rt.unet.model.dtype
    rt.whisper = rt.WhisperModel.from_pretrained(rt.args.whisper_dir)
    rt.whisper = rt.whisper.to(device=rt.device, dtype=rt.weight_dtype).eval()
    rt.whisper.requires_grad_(False)

    if rt.args.version == "v15":
        rt.fp = rt.FaceParsing(
            left_cheek_width=rt.args.left_cheek_width,
            right_cheek_width=rt.args.right_cheek_width,
        )
    else:
        rt.fp = rt.FaceParsing()


def wait_for_stable_file(path: Path, settle_ms: int) -> bool:
    if not path.exists():
        return False
    before = path.stat()
    time.sleep(settle_ms / 1000.0)
    if not path.exists():
        return False
    after = path.stat()
    return (
        before.st_mtime_ns == after.st_mtime_ns
        and before.st_size == after.st_size
        and after.st_size > 44
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuseTalk worker that watches a rolling wav chunk and runs prepared-avatar inference."
    )
    parser.add_argument("--avatar-id", type=str, required=True, help="Prepared avatar id")
    parser.add_argument(
        "--chunk-wav",
        type=str,
        default="data/audio/live/latest_chunk.wav",
        help="Rolling wav produced by personaplex_audio_bridge.py",
    )
    parser.add_argument("--poll-ms", type=int, default=120, help="Chunk file poll interval")
    parser.add_argument("--settle-ms", type=int, default=60, help="Wait this long before reading changed chunk")
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means infinite")
    parser.add_argument("--output-prefix", type=str, default="live", help="Output mp4 prefix")
    parser.add_argument(
        "--latest-jpeg",
        type=str,
        default="data/video/live/latest.jpg",
        help="Write the latest generated frame here for web streaming servers.",
    )

    parser.add_argument("--version", type=str, default="v15", choices=["v15", "v1"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--vae-type", type=str, default="sd-vae")
    parser.add_argument("--unet-config", type=str, default="models/musetalkV15/musetalk.json")
    parser.add_argument("--unet-model-path", type=str, default="models/musetalkV15/unet.pth")
    parser.add_argument("--whisper-dir", type=str, default="models/whisper")
    parser.add_argument("--ffmpeg-path", type=str, default="tools/ffmpeg")
    parser.add_argument("--bbox-shift", type=int, default=0)
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--audio-padding-length-left", type=int, default=2)
    parser.add_argument("--audio-padding-length-right", type=int, default=2)
    parser.add_argument("--parsing-mode", type=str, default="jaw")
    parser.add_argument("--extra-margin", type=int, default=10)
    parser.add_argument("--left-cheek-width", type=int, default=90)
    parser.add_argument("--right-cheek-width", type=int, default=90)
    parser.add_argument("--skip-save-images", action="store_true")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--require-mmpose", action="store_true")
    parser.add_argument(
        "--publish-fps",
        type=float,
        default=12.0,
        help="JPEG publish fps for web stream playback from generated mp4 frames.",
    )
    parser.add_argument(
        "--idle-log-sec",
        type=float,
        default=5.0,
        help="How often to print waiting/debug status when no new chunk is processed.",
    )
    parser.add_argument(
        "--status-json",
        type=str,
        default="data/video/live/worker_status.json",
        help="Path to write rolling worker status for debugging.",
    )

    return parser.parse_args()


def extract_latest_frame(mp4_path: Path, latest_jpeg: Path) -> bool:
    # Try OpenCV first; it is less brittle than ffmpeg seek for very short clips.
    cap = cv2.VideoCapture(str(mp4_path))
    if cap.isOpened():
        ok = False
        frame = None
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            ok = True
            frame = frm
        cap.release()
        if ok and frame is not None:
            tmp = latest_jpeg.with_suffix(".tmp.jpg")
            cv2.imwrite(str(tmp), frame)
            if tmp.exists():
                tmp.replace(latest_jpeg)
                return True

    # Fallback to ffmpeg frame extraction.
    tmp_jpg = latest_jpeg.with_suffix(".tmp.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(mp4_path),
        "-frames:v",
        "1",
        str(tmp_jpg),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode == 0 and tmp_jpg.exists():
        tmp_jpg.replace(latest_jpeg)
        return True
    if proc.stderr:
        print(f"[worker][warn] ffmpeg extract failed: {proc.stderr.strip()}")
    return False


def publish_mp4_frames(mp4_path: Path, latest_jpeg: Path, publish_fps: float) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return 0
    tmp = latest_jpeg.with_suffix(".tmp.jpg")
    frame_count = 0
    delay = 1.0 / max(1.0, float(publish_fps))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(tmp), frame)
        if tmp.exists():
            tmp.replace(latest_jpeg)
            frame_count += 1
        time.sleep(delay)
    cap.release()
    return frame_count


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    args = parse_args()
    setup_runtime(args)
    rt = get_rt()

    avatar = rt.Avatar(
        avatar_id=args.avatar_id,
        video_path=str(args.chunk_wav),
        bbox_shift=args.bbox_shift,
        batch_size=args.batch_size,
        preparation=False,
    )
    chunk_path = Path(args.chunk_wav).expanduser().resolve()
    latest_jpeg = Path(args.latest_jpeg).expanduser().resolve()
    status_json = Path(args.status_json).expanduser().resolve()
    latest_jpeg.parent.mkdir(parents=True, exist_ok=True)
    status_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"[worker] watching chunk wav: {chunk_path}")
    print(f"[worker] avatar: {args.avatar_id}")
    if args.skip_save_images:
        print("[worker][warn] --skip-save-images is ON; mp4 output may be missing, stream will stay black.")
    last_sig = None
    last_seen_chunk_sig = None
    jobs = 0
    last_log_t = 0.0
    last_chunk_processed_t = 0.0
    last_jpeg_update_t = 0.0
    last_mp4_path = ""

    def write_status(last_error: str = "") -> None:
        payload = {
            "avatar_id": args.avatar_id,
            "chunk_wav": str(chunk_path),
            "latest_jpeg": str(latest_jpeg),
            "jobs_done": jobs,
            "last_chunk_processed_epoch": last_chunk_processed_t or None,
            "last_jpeg_update_epoch": last_jpeg_update_t or None,
            "last_output_mp4": last_mp4_path or None,
            "chunk_exists": chunk_path.exists(),
            "jpeg_exists": latest_jpeg.exists(),
            "last_error": last_error or None,
        }
        tmp = status_json.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(status_json)

    while True:
        write_status()
        if chunk_path.exists():
            stat = chunk_path.stat()
            sig = (stat.st_mtime_ns, stat.st_size)
            last_seen_chunk_sig = sig
            if sig != last_sig and wait_for_stable_file(chunk_path, settle_ms=args.settle_ms):
                print(f"[worker] new chunk detected: bytes={stat.st_size}")
                out_name = f"{args.output_prefix}_{jobs:06d}"
                print(f"[worker] infer chunk -> {out_name}.mp4")
                avatar.inference(
                    str(chunk_path),
                    out_name,
                    args.fps,
                    args.skip_save_images,
                )
                output_mp4 = Path(avatar.video_out_path) / f"{out_name}.mp4"
                last_mp4_path = str(output_mp4)
                last_chunk_processed_t = time.time()
                if output_mp4.exists():
                    n_frames = publish_mp4_frames(output_mp4, latest_jpeg, args.publish_fps)
                    if n_frames > 0:
                        print(
                            f"[worker] published {n_frames} frames to stream at "
                            f"{args.publish_fps:.1f} fps: {latest_jpeg}"
                        )
                        last_jpeg_update_t = time.time()
                    else:
                        ok = extract_latest_frame(output_mp4, latest_jpeg)
                        if ok:
                            print(f"[worker] fallback single-frame update: {latest_jpeg}")
                            last_jpeg_update_t = time.time()
                        else:
                            print(f"[worker][warn] could not extract frame from: {output_mp4}")
                        write_status(last_error=f"frame_extract_failed:{output_mp4}")
                else:
                    print(f"[worker][warn] expected output missing: {output_mp4}")
                    write_status(last_error=f"missing_output_mp4:{output_mp4}")
                last_sig = sig
                jobs += 1
                if args.max_jobs > 0 and jobs >= args.max_jobs:
                    print("[worker] max jobs reached, exiting.")
                    break
        now = time.time()
        if now - last_log_t >= max(1.0, args.idle_log_sec):
            if not chunk_path.exists():
                print(f"[worker][debug] waiting: chunk file not found yet -> {chunk_path}")
            elif last_seen_chunk_sig == last_sig:
                print(f"[worker][debug] waiting: no new chunk update (last bytes={last_sig[1] if last_sig else 'n/a'})")
            elif latest_jpeg.exists():
                j = latest_jpeg.stat()
                print(f"[worker][debug] jpeg present size={j.st_size} mtime_ns={j.st_mtime_ns}")
            else:
                print(f"[worker][debug] jpeg not created yet: {latest_jpeg}")
            last_log_t = now
        time.sleep(max(args.poll_ms, 20) / 1000.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[worker] stopped by user")
