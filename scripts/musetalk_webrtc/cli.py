"""CLI entrypoints for the modular MuseTalk WebRTC server."""

import argparse
from pathlib import Path

from aiohttp import web

from .constants import SESSION_TOKEN_HEADER
from .models import AppArgs
from .rtc import AIORTC_AVAILABLE
from .server import WebRtcApp

def parse_args() -> AppArgs:
    """Parse CLI flags and return strongly typed `AppArgs`.

    Receives:
    - Process command-line arguments.

    Returns:
    - `AppArgs` containing all runtime configuration.
    """

    parser = argparse.ArgumentParser(
        description="In-memory PersonaPlex -> MuseTalk -> WebRTC pipeline (no WAV/MP4 loop)."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument(
        "--ice-server",
        action="append",
        default=["stun:stun.l.google.com:19302"],
        help="ICE server URL. Repeat flag for multiple entries (stun:..., turn:..., turns:...).",
    )
    parser.add_argument(
        "--ice-transport-policy",
        type=str,
        default="all",
        choices=["all", "relay"],
        help="Use 'relay' to force TURN relay candidates only.",
    )
    parser.add_argument("--ice-username", type=str, default="", help="ICE username for TURN auth.")
    parser.add_argument("--ice-credential", type=str, default="", help="ICE credential/password for TURN auth.")

    parser.add_argument("--personaplex-host", type=str, default="127.0.0.1")
    parser.add_argument("--personaplex-port", type=int, default=8998)
    parser.add_argument("--personaplex-path", type=str, default="/api/avatar/audio")
    parser.add_argument("--personaplex-text-prompt", type=str, default="")
    parser.add_argument("--personaplex-voice-prompt", type=str, default="")
    parser.add_argument("--personaplex-extra-query", action="append", default=[])

    parser.add_argument("--avatar-id", type=str, required=True)
    parser.add_argument("--version", type=str, default="v15", choices=["v15", "v1"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--require-mmpose", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bbox-shift", type=int, default=0)
    parser.add_argument("--unet-model-path", type=str, default="models/musetalkV15/unet.pth")
    parser.add_argument("--unet-config", type=str, default="models/musetalkV15/musetalk.json")
    parser.add_argument("--vae-type", type=str, default="sd-vae")
    parser.add_argument("--whisper-dir", type=str, default="models/whisper")
    parser.add_argument("--ffmpeg-path", type=str, default="tools/ffmpeg")
    parser.add_argument("--parsing-mode", type=str, default="jaw")
    parser.add_argument("--extra-margin", type=int, default=10)
    parser.add_argument("--left-cheek-width", type=int, default=90)
    parser.add_argument("--right-cheek-width", type=int, default=90)
    parser.add_argument("--audio-padding-length-left", type=int, default=2)
    parser.add_argument("--audio-padding-length-right", type=int, default=2)

    parser.add_argument("--ring-buffer-seconds", type=float, default=8.0)
    parser.add_argument("--window-ms", type=int, default=400)
    parser.add_argument("--hop-ms", type=int, default=60)
    parser.add_argument("--min-window-ms", type=int, default=240)
    parser.add_argument("--max-advance-ms", type=int, default=180)
    parser.add_argument("--max-tail-frames", type=int, default=3)
    parser.add_argument(
        "--mouth-smoothing-alpha",
        type=float,
        default=0.7,
        help="0..1. Lower values smooth more but can reduce lip sharpness.",
    )
    parser.add_argument("--video-queue-size", type=int, default=256)
    parser.add_argument("--reconnect-delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--input-source",
        type=str,
        default="mirror",
        choices=["mirror", "webrtc", "mixed"],
        help="Audio input path for driving MuseTalk: mirror websocket, WebRTC uplink, or both.",
    )
    parser.add_argument(
        "--webrtc-audio-loopback",
        action="store_true",
        help="Loop inbound WebRTC mic audio back to outbound avatar audio track for MVP testing.",
    )
    parser.add_argument(
        "--musetalk-only",
        action="store_true",
        help="Disable PersonaPlex I/O and drive lipsync directly from inbound WebRTC mic audio.",
    )
    parser.add_argument(
        "--enable-api-auth",
        action="store_true",
        help="Scaffold auth for /v1 endpoints. Disabled by default.",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default="",
        help="Bearer token used when --enable-api-auth is set.",
    )
    parser.add_argument(
        "--session-offer-timeout-seconds",
        type=float,
        default=90.0,
        help="Expire sessions that never receive SDP offer after this timeout.",
    )
    parser.add_argument(
        "--session-max-age-seconds",
        type=float,
        default=7200.0,
        help="Force-expire sessions older than this max age.",
    )
    parser.add_argument(
        "--session-cleanup-interval-seconds",
        type=float,
        default=5.0,
        help="Background interval for expiring stale sessions.",
    )
    parser.add_argument(
        "--multi-session",
        action="store_true",
        help="Allow multiple concurrent sessions (disabled by default).",
    )
    parser.add_argument(
        "--web-test-only",
        action="store_true",
        help="Run signaling/web app only (no MuseTalk model load). Useful for WebRTC connection testing.",
    )
    parser.add_argument(
        "--status-json",
        type=str,
        default="data/live/in_memory_pipeline_status.json",
        help="Optional status file path. Empty string disables.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose runtime debug logs and include debug events in /status.",
    )
    parser.add_argument(
        "--debug-events-limit",
        type=int,
        default=250,
        help="Maximum in-memory debug events retained for /status.",
    )

    ns = parser.parse_args()
    status_json = Path(ns.status_json).expanduser().resolve() if str(ns.status_json).strip() else None
    return AppArgs(
        host=ns.host,
        port=ns.port,
        ice_servers=ns.ice_server,
        ice_transport_policy=ns.ice_transport_policy,
        ice_username=ns.ice_username,
        ice_credential=ns.ice_credential,
        personaplex_host=ns.personaplex_host,
        personaplex_port=ns.personaplex_port,
        personaplex_path=ns.personaplex_path,
        personaplex_text_prompt=ns.personaplex_text_prompt,
        personaplex_voice_prompt=ns.personaplex_voice_prompt,
        personaplex_extra_query=ns.personaplex_extra_query,
        avatar_id=ns.avatar_id,
        version=ns.version,
        gpu_id=ns.gpu_id,
        use_fp16=ns.use_fp16,
        require_mmpose=ns.require_mmpose,
        fps=ns.fps,
        batch_size=ns.batch_size,
        bbox_shift=ns.bbox_shift,
        unet_model_path=ns.unet_model_path,
        unet_config=ns.unet_config,
        vae_type=ns.vae_type,
        whisper_dir=ns.whisper_dir,
        ffmpeg_path=ns.ffmpeg_path,
        parsing_mode=ns.parsing_mode,
        extra_margin=ns.extra_margin,
        left_cheek_width=ns.left_cheek_width,
        right_cheek_width=ns.right_cheek_width,
        audio_padding_length_left=ns.audio_padding_length_left,
        audio_padding_length_right=ns.audio_padding_length_right,
        ring_buffer_seconds=ns.ring_buffer_seconds,
        window_ms=ns.window_ms,
        hop_ms=ns.hop_ms,
        min_window_ms=ns.min_window_ms,
        max_advance_ms=ns.max_advance_ms,
        max_tail_frames=ns.max_tail_frames,
        mouth_smoothing_alpha=ns.mouth_smoothing_alpha,
        video_queue_size=ns.video_queue_size,
        status_json=status_json,
        reconnect_delay_seconds=ns.reconnect_delay_seconds,
        input_source=ns.input_source,
        webrtc_audio_loopback=ns.webrtc_audio_loopback,
        musetalk_only=ns.musetalk_only,
        enable_api_auth=ns.enable_api_auth,
        api_token=ns.api_token,
        session_offer_timeout_seconds=ns.session_offer_timeout_seconds,
        session_max_age_seconds=ns.session_max_age_seconds,
        session_cleanup_interval_seconds=ns.session_cleanup_interval_seconds,
        single_session_mode=(not ns.multi_session),
        web_test_only=ns.web_test_only,
        debug=ns.debug,
        debug_events_limit=ns.debug_events_limit,
    )


def main():
    """Program entry point: validate deps, build app, and run server."""

    args = parse_args()
    if not AIORTC_AVAILABLE:
        raise SystemExit("aiortc/av is required for WebRTC output. Install with: pip install aiortc av")
    chat_mode = (not args.web_test_only) and args.personaplex_path.rstrip("/").endswith("/api/chat")
    if chat_mode and not str(args.personaplex_voice_prompt).strip():
        print(
            "[webrtc][warn] personaplex_path=/api/chat but --personaplex-voice-prompt is empty. "
            "If Personaplex enforces voice_prompt_dir, set a valid filename (e.g. s0.wav)."
        )
    app_state = WebRtcApp(args)
    app = app_state.build_app()
    print(f"[webrtc] serving http://{args.host}:{args.port}/")
    print(
        "[webrtc] endpoints: /healthz, /offer (legacy), /v1/config, /v1/session, "
        "/v1/sessions (create), /v1/sessions/{id}/offer, /v1/sessions/{id}/stats, /status"
    )
    print(f"[webrtc] input_source={args.input_source} webrtc_audio_loopback={args.webrtc_audio_loopback}")
    if args.musetalk_only:
        print("[webrtc] musetalk-only mode enabled (no PersonaPlex connections).")
    print(f"[webrtc] single_session_mode={args.single_session_mode} session_token_header={SESSION_TOKEN_HEADER}")
    print(f"[webrtc] personaplex_path={args.personaplex_path} chat_mode={chat_mode}")
    if args.web_test_only:
        print("[webrtc] web-test-only enabled (MuseTalk model initialization skipped)")
    if args.enable_api_auth:
        print("[webrtc] /v1 auth middleware enabled")
    if args.debug:
        print(f"[webrtc] debug mode enabled (events_limit={args.debug_events_limit})")
    web.run_app(app, host=args.host, port=args.port)
