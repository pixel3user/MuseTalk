"""Backward-compatible entrypoint for the modular MuseTalk WebRTC server."""

try:
    from .musetalk_webrtc.cli import main
except ImportError as e:  # pragma: no cover - allows direct execution as a script.
    # Only fallback for direct script execution context.
    if "attempted relative import with no known parent package" not in str(e):
        raise
    from scripts.musetalk_webrtc.cli import main


if __name__ == "__main__":
    main()
