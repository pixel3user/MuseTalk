import argparse
import asyncio
import contextlib
import time
from pathlib import Path

from aiohttp import web


HTML_PAGE = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Avatar Stream</title></head>
<body style="margin:0;background:#111;color:#ddd;font-family:sans-serif">
  <div style="padding:10px">MuseTalk Avatar Stream</div>
  <img src="/mjpeg" style="width:100vw;height:calc(100vh - 40px);object-fit:contain;" />
</body>
</html>
"""


class AvatarStreamServer:
    def __init__(self, latest_jpeg: Path, fps: int, log_interval_sec: float):
        self.latest_jpeg = latest_jpeg
        self.frame_interval = 1.0 / max(1, fps)
        self.log_interval_sec = max(1.0, log_interval_sec)
        self.active_clients = 0
        self.total_frames_sent = 0
        self.last_frame_sent_at = 0.0
        self.last_stream_log_at = 0.0
        self.last_stream_sig = None

    def _jpeg_state(self) -> dict:
        if not self.latest_jpeg.exists():
            return {
                "exists": False,
                "size_bytes": 0,
                "mtime_ns": 0,
                "path": str(self.latest_jpeg),
            }
        st = self.latest_jpeg.stat()
        return {
            "exists": True,
            "size_bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "path": str(self.latest_jpeg),
        }

    async def index(self, _request: web.Request) -> web.Response:
        return web.Response(text=HTML_PAGE, content_type="text/html")

    async def status(self, _request: web.Request) -> web.Response:
        now = time.time()
        payload = {
            "active_clients": self.active_clients,
            "total_frames_sent": self.total_frames_sent,
            "last_frame_sent_epoch": self.last_frame_sent_at,
            "seconds_since_last_frame_sent": (now - self.last_frame_sent_at) if self.last_frame_sent_at else None,
            "latest_jpeg": self._jpeg_state(),
        }
        return web.json_response(payload)

    async def mjpeg(self, request: web.Request) -> web.StreamResponse:
        boundary = "frame"
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        self.active_clients += 1
        last_sig = None
        try:
            while True:
                jpeg_state = self._jpeg_state()
                if jpeg_state["exists"]:
                    sig = (jpeg_state["mtime_ns"], jpeg_state["size_bytes"])
                    if sig != last_sig and jpeg_state["size_bytes"] > 0:
                        data = self.latest_jpeg.read_bytes()
                        part_header = (
                            f"--{boundary}\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(data)}\r\n\r\n"
                        ).encode("ascii")
                        await resp.write(part_header + data + b"\r\n")
                        last_sig = sig
                        self.last_stream_sig = sig
                        self.total_frames_sent += 1
                        self.last_frame_sent_at = time.time()
                now = time.time()
                if now - self.last_stream_log_at >= self.log_interval_sec:
                    if not jpeg_state["exists"]:
                        print(f"[mjpeg][debug] waiting for frame file: {self.latest_jpeg}")
                    elif jpeg_state["size_bytes"] <= 0:
                        print(f"[mjpeg][debug] frame file is empty: {self.latest_jpeg}")
                    elif self.last_stream_sig != (jpeg_state["mtime_ns"], jpeg_state["size_bytes"]):
                        print(
                            f"[mjpeg][debug] frame exists but not emitted yet "
                            f"(size={jpeg_state['size_bytes']})"
                        )
                    self.last_stream_log_at = now
                await asyncio.sleep(self.frame_interval)
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self.active_clients = max(0, self.active_clients - 1)
            with contextlib.suppress(Exception):
                await resp.write_eof()
        return resp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve latest MuseTalk frame as MJPEG stream.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--latest-jpeg",
        type=str,
        default="data/video/live/latest.jpg",
        help="Path written by musetalk_chunk_worker --latest-jpeg",
    )
    parser.add_argument("--fps", type=int, default=15, help="MJPEG push fps cap")
    parser.add_argument("--log-interval-sec", type=float, default=5.0, help="Debug log cadence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latest_jpeg = Path(args.latest_jpeg).expanduser().resolve()
    latest_jpeg.parent.mkdir(parents=True, exist_ok=True)
    server = AvatarStreamServer(latest_jpeg=latest_jpeg, fps=args.fps, log_interval_sec=args.log_interval_sec)

    app = web.Application()
    app.router.add_get("/", server.index)
    app.router.add_get("/mjpeg", server.mjpeg)
    app.router.add_get("/status", server.status)
    print(f"[mjpeg] serving on http://{args.host}:{args.port}/ (stream at /mjpeg)")
    print(f"[mjpeg] latest frame path: {latest_jpeg}")
    print(f"[mjpeg] status endpoint: http://{args.host}:{args.port}/status")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
