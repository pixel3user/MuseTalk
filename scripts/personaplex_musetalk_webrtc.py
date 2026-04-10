"""Backward-compatible entrypoint for the modular MuseTalk WebRTC server."""

try:
    from .musetalk_webrtc.cli import main
except Exception:  # pragma: no cover - allows direct execution as a script.
    from musetalk_webrtc.cli import main


if __name__ == "__main__":
    main()
