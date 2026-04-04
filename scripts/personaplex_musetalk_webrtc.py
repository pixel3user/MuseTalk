import argparse
import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import aiohttp
import cv2
import librosa
import numpy as np
import sphn
import torch
from aiohttp import web

try:
    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack

    AIORTC_AVAILABLE = True
except Exception:
    AIORTC_AVAILABLE = False
    av = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    AudioStreamTrack = object
    VideoStreamTrack = object


HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MuseTalk WebRTC</title>
</head>
<body style="margin:0;background:#121212;color:#e5e5e5;font-family:system-ui,sans-serif">
  <div style="padding:12px;font-size:14px">MuseTalk In-Memory WebRTC Preview</div>
  <div style="display:flex;gap:12px;padding:0 12px 12px 12px;flex-wrap:wrap">
    <video id="v" autoplay playsinline muted style="width:min(96vw,960px);background:black;border-radius:10px"></video>
    <audio id="a" autoplay controls style="width:min(96vw,960px)"></audio>
  </div>
  <div style="padding:0 12px 12px 12px">
    <button id="start">Start</button>
    <span id="state" style="margin-left:8px">idle</span>
  </div>
<script>
let pc = null;
async function start() {
  const state = document.getElementById('state');
  state.textContent = 'connecting';
  pc = new RTCPeerConnection();
  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });
  pc.ontrack = (ev) => {
    if (ev.track.kind === 'video') {
      document.getElementById('v').srcObject = ev.streams[0];
    } else if (ev.track.kind === 'audio') {
      document.getElementById('a').srcObject = ev.streams[0];
    }
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const resp = await fetch('/offer', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
  });
  const answer = await resp.json();
  await pc.setRemoteDescription(answer);
  state.textContent = 'connected';
}
document.getElementById('start').onclick = start;
</script>
</body>
</html>
"""


@dataclass
class AppArgs:
    host: str
    port: int
    personaplex_host: str
    personaplex_port: int
    personaplex_path: str
    personaplex_text_prompt: str
    personaplex_voice_prompt: str
    personaplex_extra_query: list[str]
    avatar_id: str
    version: str
    gpu_id: int
    use_fp16: bool
    require_mmpose: bool
    fps: int
    batch_size: int
    bbox_shift: int
    unet_model_path: str
    unet_config: str
    vae_type: str
    whisper_dir: str
    ffmpeg_path: str
    parsing_mode: str
    extra_margin: int
    left_cheek_width: int
    right_cheek_width: int
    audio_padding_length_left: int
    audio_padding_length_right: int
    ring_buffer_seconds: float
    window_ms: int
    hop_ms: int
    min_window_ms: int
    video_queue_size: int
    status_json: Optional[Path]
    reconnect_delay_seconds: float


class PcmRingBuffer:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.buf = np.zeros((0,), dtype=np.float32)
        self.total_samples = 0
        self.lock = asyncio.Lock()

    async def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        async with self.lock:
            self.total_samples += int(samples.size)
            self.buf = np.concatenate([self.buf, samples.astype(np.float32, copy=False)])
            if self.buf.size > self.max_samples:
                self.buf = self.buf[-self.max_samples :]

    async def latest(self, n_samples: int) -> tuple[np.ndarray, int]:
        async with self.lock:
            if self.buf.size <= n_samples:
                return self.buf.copy(), self.total_samples
            return self.buf[-n_samples:].copy(), self.total_samples


class AudioTrackBuffer:
    def __init__(self, max_samples_48k: int):
        self.max_samples = max_samples_48k
        self.buf = np.zeros((0,), dtype=np.float32)
        self.lock = asyncio.Lock()

    async def append_from_24k(self, mono24k: np.ndarray) -> None:
        if mono24k.size == 0:
            return
        mono48k = np.repeat(mono24k.astype(np.float32, copy=False), 2)
        async with self.lock:
            self.buf = np.concatenate([self.buf, mono48k])
            if self.buf.size > self.max_samples:
                self.buf = self.buf[-self.max_samples :]

    async def pop_48k(self, n_samples: int) -> np.ndarray:
        async with self.lock:
            if self.buf.size >= n_samples:
                out = self.buf[:n_samples].copy()
                self.buf = self.buf[n_samples:]
                return out
            if self.buf.size == 0:
                return np.zeros((n_samples,), dtype=np.float32)
            out = np.zeros((n_samples,), dtype=np.float32)
            out[: self.buf.size] = self.buf
            self.buf = np.zeros((0,), dtype=np.float32)
            return out


class VideoFrameBuffer:
    def __init__(self, maxsize: int):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.last_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    async def publish(self, frame_bgr: np.ndarray) -> None:
        self.last_frame = frame_bgr
        if self.queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                _ = self.queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(frame_bgr)

    async def get(self, timeout: float = 0.12) -> np.ndarray:
        try:
            frame = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            self.last_frame = frame
            return frame
        except asyncio.TimeoutError:
            return self.last_frame


class PersonaPlexMirrorClient:
    def __init__(
        self,
        ws_url: str,
        pcm_ring_24k: PcmRingBuffer,
        audio_track_buffer: AudioTrackBuffer,
        status_json: Optional[Path],
        reconnect_delay_seconds: float,
    ):
        self.ws_url = ws_url
        self.ring = pcm_ring_24k
        self.audio_track_buffer = audio_track_buffer
        self.reader = sphn.OpusStreamReader(24000)
        self.status_json = status_json
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.stop_event = asyncio.Event()
        self.packets = 0
        self.decoded_seconds = 0.0
        self.last_rx_epoch = 0.0
        self.connected = False

    def _write_status(self) -> None:
        if self.status_json is None:
            return
        payload = {
            "ws_url": self.ws_url,
            "connected": self.connected,
            "audio_packets": self.packets,
            "decoded_seconds": self.decoded_seconds,
            "last_audio_rx_epoch": self.last_rx_epoch or None,
        }
        self.status_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_json.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.status_json)

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            while not self.stop_event.is_set():
                try:
                    print(f"[mirror] connecting: {self.ws_url}")
                    async with session.ws_connect(self.ws_url, heartbeat=None) as ws:
                        self.connected = True
                        self._write_status()
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.BINARY:
                                continue
                            data = msg.data
                            if not isinstance(data, (bytes, bytearray)) or len(data) < 2:
                                continue
                            kind = data[0]
                            payload = bytes(data[1:])
                            if kind == 0:
                                print("[mirror] handshake received")
                                continue
                            if kind != 1:
                                continue
                            self.packets += 1
                            self.reader.append_bytes(payload)
                            pcm = self.reader.read_pcm()
                            if pcm.shape[-1] == 0:
                                continue
                            pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
                            await self.ring.append(pcm)
                            await self.audio_track_buffer.append_from_24k(pcm)
                            self.decoded_seconds += pcm.size / 24000.0
                            self.last_rx_epoch = time.time()
                            self._write_status()
                except Exception as e:
                    self.connected = False
                    self._write_status()
                    print(f"[mirror] connection error: {e!r}")
                    await asyncio.sleep(max(0.2, self.reconnect_delay_seconds))


class MuseTalkRealtimeEngine:
    def __init__(
        self,
        args: AppArgs,
        pcm_ring_24k: PcmRingBuffer,
        video_buffer: VideoFrameBuffer,
    ):
        self.args = args
        self.pcm_ring = pcm_ring_24k
        self.video_buffer = video_buffer
        self.stop_event = asyncio.Event()
        self.last_total_samples = -1
        self.avatar_frame_idx = 0
        self.jobs = 0
        self.last_publish_epoch = 0.0
        self.last_error = ""

        import scripts.realtime_inference as rt

        self.rt = rt
        self._setup_runtime()
        self.avatar = self.rt.Avatar(
            avatar_id=args.avatar_id,
            video_path="unused",
            bbox_shift=args.bbox_shift,
            batch_size=args.batch_size,
            preparation=False,
        )

    def _setup_runtime(self) -> None:
        rt = self.rt
        rt.args = argparse.Namespace(
            version=self.args.version,
            ffmpeg_path=self.args.ffmpeg_path,
            gpu_id=self.args.gpu_id,
            vae_type=self.args.vae_type,
            unet_config=self.args.unet_config,
            unet_model_path=self.args.unet_model_path,
            whisper_dir=self.args.whisper_dir,
            inference_config="",
            bbox_shift=self.args.bbox_shift,
            result_dir="results",
            extra_margin=self.args.extra_margin,
            fps=self.args.fps,
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
            batch_size=self.args.batch_size,
            output_vid_name=None,
            use_saved_coord=False,
            saved_coord=False,
            parsing_mode=self.args.parsing_mode,
            left_cheek_width=self.args.left_cheek_width,
            right_cheek_width=self.args.right_cheek_width,
            skip_save_images=True,
            non_interactive=True,
            force_recreate_avatar=False,
            use_fp16=self.args.use_fp16,
            require_mmpose=self.args.require_mmpose,
        )
        rt.device = torch.device(f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")

        if rt.args.require_mmpose and not rt.MMPOSE_AVAILABLE:
            raise RuntimeError("mmpose/DWPose is required but unavailable.")
        if not rt.MMPOSE_AVAILABLE:
            print("[engine][warn] mmpose unavailable; using fallback face detector.")

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
            print("[engine] precision: fp16")
        else:
            rt.pe = rt.pe.float().to(rt.device)
            rt.vae.vae = rt.vae.vae.float().to(rt.device)
            rt.unet.model = rt.unet.model.float().to(rt.device)
            print("[engine] precision: fp32")

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

    def _infer_window_frames(self, pcm24k_window: np.ndarray, new_frames: int) -> list[np.ndarray]:
        rt = self.rt
        audio16k = librosa.resample(pcm24k_window.astype(np.float32), orig_sr=24000, target_sr=16000)
        feature_ret = rt.audio_processor.get_audio_feature_from_array(
            audio16k, sample_rate=16000, weight_dtype=rt.weight_dtype
        )
        if feature_ret is None:
            return []
        whisper_input_features, librosa_length = feature_ret
        whisper_chunks = rt.audio_processor.get_whisper_chunk(
            whisper_input_features,
            rt.device,
            rt.weight_dtype,
            rt.whisper,
            librosa_length,
            fps=self.args.fps,
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
        )
        if whisper_chunks is None or len(whisper_chunks) == 0:
            return []

        combined_frames = []
        gen = rt.datagen(whisper_chunks, self.avatar.input_latent_list_cycle, self.args.batch_size)
        for whisper_batch, latent_batch in gen:
            audio_feature_batch = rt.pe(whisper_batch.to(rt.device))
            latent_batch = latent_batch.to(device=rt.device, dtype=rt.unet.model.dtype)
            pred_latents = rt.unet.model(
                latent_batch, rt.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents = pred_latents.to(device=rt.device, dtype=rt.vae.vae.dtype)
            recon = rt.vae.decode_latents(pred_latents)
            for res_frame in recon:
                base_i = self.avatar_frame_idx % len(self.avatar.frame_list_cycle)
                bbox = self.avatar.coord_list_cycle[base_i]
                ori_frame = self.avatar.frame_list_cycle[base_i].copy()
                x1, y1, x2, y2 = bbox
                try:
                    lip = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    self.avatar_frame_idx += 1
                    continue
                mask = self.avatar.mask_list_cycle[base_i]
                mask_box = self.avatar.mask_coords_list_cycle[base_i]
                frame = rt.get_image_blending(ori_frame, lip, bbox, mask, mask_box)
                combined_frames.append(frame)
                self.avatar_frame_idx += 1

        if not combined_frames:
            return []
        if new_frames <= 0:
            return [combined_frames[-1]]
        return combined_frames[-new_frames:]

    async def run(self) -> None:
        window_samples = int((self.args.window_ms / 1000.0) * 24000)
        min_samples = int((self.args.min_window_ms / 1000.0) * 24000)
        hop_seconds = self.args.hop_ms / 1000.0
        while not self.stop_event.is_set():
            await asyncio.sleep(max(0.02, hop_seconds))
            window, total = await self.pcm_ring.latest(window_samples)
            if window.size < min_samples:
                continue
            if total == self.last_total_samples:
                continue

            if self.last_total_samples < 0:
                new_samples = int(window.size)
            else:
                new_samples = max(1, int(total - self.last_total_samples))
            self.last_total_samples = total

            # Fixes "same lips over and over" by publishing only newly advanced tail frames.
            new_frames = max(1, int(round((new_samples / 24000.0) * self.args.fps)))
            try:
                frames = await asyncio.to_thread(self._infer_window_frames, window, new_frames)
                for frame in frames:
                    await self.video_buffer.publish(frame)
                    self.last_publish_epoch = time.time()
                if frames:
                    self.jobs += 1
            except Exception as e:
                self.last_error = repr(e)
                print(f"[engine] inference error: {e!r}")

    def status(self) -> dict:
        return {
            "jobs": self.jobs,
            "last_publish_epoch": self.last_publish_epoch or None,
            "last_error": self.last_error or None,
            "avatar_frame_idx": self.avatar_frame_idx,
        }


class MuseTalkVideoTrack(VideoStreamTrack):
    def __init__(self, buffer: VideoFrameBuffer, fps: int):
        super().__init__()
        self.buffer = buffer
        self.fps = max(1, fps)
        self.frame_interval = 1.0 / self.fps

    async def recv(self):
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc/av is not installed")
        frame_bgr = await self.buffer.get(timeout=self.frame_interval)
        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


class MuseTalkAudioTrack(AudioStreamTrack):
    def __init__(self, audio_buffer: AudioTrackBuffer):
        super().__init__()
        self.audio_buffer = audio_buffer
        self.samples_per_frame = 960  # 20ms @ 48kHz

    async def recv(self):
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc/av is not installed")
        pcm = await self.audio_buffer.pop_48k(self.samples_per_frame)
        pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples_per_frame)
        frame.sample_rate = 48000
        frame.planes[0].update(pcm_i16.tobytes())
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


class WebRtcApp:
    def __init__(self, args: AppArgs):
        self.args = args
        self.pcs = set()

        ring_samples = int(args.ring_buffer_seconds * 24000)
        self.pcm_ring_24k = PcmRingBuffer(max_samples=ring_samples)
        self.audio_track_buffer = AudioTrackBuffer(max_samples_48k=int(args.ring_buffer_seconds * 48000))
        self.video_buffer = VideoFrameBuffer(maxsize=args.video_queue_size)

        ws_url = self._build_mirror_ws_url()
        self.mirror_client = PersonaPlexMirrorClient(
            ws_url=ws_url,
            pcm_ring_24k=self.pcm_ring_24k,
            audio_track_buffer=self.audio_track_buffer,
            status_json=args.status_json,
            reconnect_delay_seconds=args.reconnect_delay_seconds,
        )
        self.engine = MuseTalkRealtimeEngine(
            args=args,
            pcm_ring_24k=self.pcm_ring_24k,
            video_buffer=self.video_buffer,
        )

        self.mirror_task: Optional[asyncio.Task] = None
        self.engine_task: Optional[asyncio.Task] = None

    def _build_mirror_ws_url(self) -> str:
        path = self.args.personaplex_path
        if not path.startswith("/"):
            path = "/" + path
        query = {
            "text_prompt": self.args.personaplex_text_prompt,
            "voice_prompt": self.args.personaplex_voice_prompt,
        }
        for kv in self.args.personaplex_extra_query:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            query[k] = v
        qs = urlencode(query)
        return f"ws://{self.args.personaplex_host}:{self.args.personaplex_port}{path}?{qs}"

    async def on_startup(self, _app: web.Application):
        self.mirror_task = asyncio.create_task(self.mirror_client.run())
        self.engine_task = asyncio.create_task(self.engine.run())

    async def on_cleanup(self, _app: web.Application):
        self.mirror_client.stop_event.set()
        self.engine.stop_event.set()
        for task in (self.mirror_task, self.engine_task):
            if task is not None:
                task.cancel()
        for task in (self.mirror_task, self.engine_task):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        await asyncio.gather(*(pc.close() for pc in list(self.pcs)), return_exceptions=True)
        self.pcs.clear()

    async def index(self, _request: web.Request) -> web.Response:
        return web.Response(text=HTML_PAGE, content_type="text/html")

    async def status(self, _request: web.Request) -> web.Response:
        payload = {
            "mirror": {
                "connected": self.mirror_client.connected,
                "packets": self.mirror_client.packets,
                "decoded_seconds": self.mirror_client.decoded_seconds,
                "last_rx_epoch": self.mirror_client.last_rx_epoch or None,
            },
            "engine": self.engine.status(),
        }
        return web.json_response(payload)

    async def offer(self, request: web.Request) -> web.Response:
        if not AIORTC_AVAILABLE:
            return web.json_response(
                {"error": "aiortc/av not installed. Install: pip install aiortc av"},
                status=500,
            )
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self.pcs.discard(pc)

        video_track = MuseTalkVideoTrack(self.video_buffer, fps=self.args.fps)
        audio_track = MuseTalkAudioTrack(self.audio_track_buffer)
        pc.addTrack(video_track)
        pc.addTrack(audio_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    def build_app(self) -> web.Application:
        app = web.Application()
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        app.router.add_get("/", self.index)
        app.router.add_get("/status", self.status)
        app.router.add_post("/offer", self.offer)
        return app


def parse_args() -> AppArgs:
    parser = argparse.ArgumentParser(
        description="In-memory PersonaPlex -> MuseTalk -> WebRTC pipeline (no WAV/MP4 loop)."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)

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
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
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

    parser.add_argument("--ring-buffer-seconds", type=float, default=12.0)
    parser.add_argument("--window-ms", type=int, default=640)
    parser.add_argument("--hop-ms", type=int, default=80)
    parser.add_argument("--min-window-ms", type=int, default=320)
    parser.add_argument("--video-queue-size", type=int, default=256)
    parser.add_argument("--reconnect-delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--status-json",
        type=str,
        default="data/live/in_memory_pipeline_status.json",
        help="Optional status file path. Empty string disables.",
    )

    ns = parser.parse_args()
    status_json = Path(ns.status_json).expanduser().resolve() if str(ns.status_json).strip() else None
    return AppArgs(
        host=ns.host,
        port=ns.port,
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
        video_queue_size=ns.video_queue_size,
        status_json=status_json,
        reconnect_delay_seconds=ns.reconnect_delay_seconds,
    )


def main():
    args = parse_args()
    if not AIORTC_AVAILABLE:
        raise SystemExit("aiortc/av is required for WebRTC output. Install with: pip install aiortc av")
    app_state = WebRtcApp(args)
    app = app_state.build_app()
    print(f"[webrtc] serving http://{args.host}:{args.port}/")
    print("[webrtc] endpoint /offer for SDP exchange, /status for diagnostics")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
