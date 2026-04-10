import argparse
import asyncio
import contextlib
import fractions
import json
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp
import cv2
import numpy as np
import sphn
import torch
from aiohttp import web
from scipy.signal import resample_poly

try:
    import av
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError, VideoStreamTrack

    AIORTC_AVAILABLE = True
except Exception:
    AIORTC_AVAILABLE = False
    av = None
    RTCConfiguration = None
    RTCIceServer = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    AudioStreamTrack = object
    MediaStreamError = RuntimeError
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
    <button id="stop" disabled>Stop</button>
    <span id="state" style="margin-left:8px">idle</span>
    <div id="mic" style="margin-top:8px;font-size:12px;color:#8fd48f">mic: idle</div>
    <div id="dbg" style="margin-top:8px;font-size:12px;white-space:pre-wrap;color:#b0b0b0"></div>
  </div>
<script>
let pc = null;
let localStream = null;
let sessionId = null;
let sessionToken = null;
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const stateEl = document.getElementById('state');
const micEl = document.getElementById('mic');

function setState(next) {
  stateEl.textContent = next;
}

function setMic(next) {
  micEl.textContent = 'mic: ' + next;
}

async function waitIceGatheringComplete(pc, timeoutMs = 4000) {
  if (pc.iceGatheringState === 'complete') return;
  await new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      pc.removeEventListener('icegatheringstatechange', onState);
      resolve();
    };
    const onState = () => {
      if (pc.iceGatheringState === 'complete') finish();
    };
    pc.addEventListener('icegatheringstatechange', onState);
    setTimeout(finish, timeoutMs);
  });
}

async function cleanupPeer() {
  if (pc) {
    try { pc.ontrack = null; } catch (e) {}
    try { pc.onconnectionstatechange = null; } catch (e) {}
    try { pc.close(); } catch (e) {}
    pc = null;
  }
  if (localStream) {
    for (const track of localStream.getTracks()) {
      try { track.stop(); } catch (e) {}
    }
    localStream = null;
  }
}

async function cleanupSession() {
  if (!sessionId || !sessionToken) {
    sessionId = null;
    sessionToken = null;
    return;
  }
  try {
    await fetch('/v1/sessions/' + sessionId, {
      method: 'DELETE',
      headers: { 'x-session-token': sessionToken },
    });
  } catch (e) {}
  sessionId = null;
  sessionToken = null;
}

async function start() {
  if (pc) {
    await stop();
  }
  startBtn.disabled = true;
  stopBtn.disabled = false;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Browser does not support getUserMedia');
    }
    setState('requesting-mic');
    setMic('requesting permission');

    const rtcCfg = await (await fetch('/config')).json();
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
    setMic('granted');

    pc = new RTCPeerConnection(rtcCfg);
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    for (const track of localStream.getAudioTracks()) {
      pc.addTrack(track, localStream);
    }
    pc.onconnectionstatechange = () => {
      if (pc) setState(pc.connectionState);
    };
    pc.ontrack = (ev) => {
      if (ev.track.kind === 'video') {
        document.getElementById('v').srcObject = ev.streams[0];
      } else if (ev.track.kind === 'audio') {
        document.getElementById('a').srcObject = ev.streams[0];
      }
    };

    setState('connecting');
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    // Non-trickle flow: wait for ICE gathering so TURN/relay candidates are included.
    await waitIceGatheringComplete(pc, 5000);
    const resp = await fetch('/offer', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!resp.ok) {
      throw new Error('offer failed: ' + await resp.text());
    }
    const answer = await resp.json();
    sessionId = answer.session_id || null;
    sessionToken = answer.session_token || null;
    await pc.setRemoteDescription(answer);
    setState('connected');
  } catch (e) {
    console.error(e);
    setState('error');
    setMic('error');
    await cleanupPeer();
    await cleanupSession();
    stopBtn.disabled = true;
  } finally {
    startBtn.disabled = false;
  }
}

async function stop() {
  stopBtn.disabled = true;
  setState('stopping');
  await cleanupPeer();
  await cleanupSession();
  document.getElementById('v').srcObject = null;
  document.getElementById('a').srcObject = null;
  setMic('idle');
  setState('idle');
}

startBtn.onclick = start;
stopBtn.onclick = stop;

setInterval(async () => {
  try {
    const s = await (await fetch('/status')).json();
    document.getElementById('dbg').textContent = JSON.stringify(s, null, 2);
  } catch (e) {}
}, 1000);

window.addEventListener('beforeunload', () => {
  try {
    if (localStream) {
      for (const track of localStream.getTracks()) {
        track.stop();
      }
    }
    if (pc) {
      pc.close();
    }
  } catch (e) {}
});
</script>
</body>
</html>
"""

SESSION_TOKEN_HEADER = "x-session-token"


@dataclass
class AppArgs:
    host: str
    port: int
    ice_servers: list[str]
    ice_transport_policy: str
    ice_username: str
    ice_credential: str
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
    max_advance_ms: int
    max_tail_frames: int
    mouth_smoothing_alpha: float
    video_queue_size: int
    status_json: Optional[Path]
    reconnect_delay_seconds: float
    input_source: str
    webrtc_audio_loopback: bool
    enable_api_auth: bool
    api_token: str
    session_offer_timeout_seconds: float
    session_max_age_seconds: float
    session_cleanup_interval_seconds: float
    single_session_mode: bool
    web_test_only: bool


@dataclass
class SessionState:
    session_id: str
    token: str
    created_epoch: float
    last_activity_epoch: float
    pc: Optional["RTCPeerConnection"] = None
    pcid: Optional[str] = None
    inbound_task: Optional[asyncio.Task] = None
    mic_frames_rx: int = 0
    mic_samples_rx_16k: int = 0
    last_mic_rx_epoch: float = 0.0
    last_offer_epoch: float = 0.0
    personaplex_audio_frames_tx: int = 0
    personaplex_audio_frames_rx: int = 0
    personaplex_last_rx_epoch: float = 0.0
    personaplex_connected: bool = False
    personaplex_bridge: Optional["PersonaPlexChatBridge"] = None
    personaplex_bridge_task: Optional[asyncio.Task] = None
    close_reason: str = ""


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

    def snapshot_jpeg(self) -> bytes:
        ok, enc = cv2.imencode(".jpg", self.last_frame)
        if not ok:
            return b""
        return enc.tobytes()


class PersonaPlexMirrorClient:
    def __init__(
        self,
        ws_url: str,
        pcm_ring_24k: PcmRingBuffer,
        pcm_ring_16k: PcmRingBuffer,
        audio_track_buffer: AudioTrackBuffer,
        status_json: Optional[Path],
        reconnect_delay_seconds: float,
    ):
        self.ws_url = ws_url
        self.ring = pcm_ring_24k
        self.ring16k = pcm_ring_16k
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
                            # Fast incremental 24k -> 16k conversion (2/3) for inference path.
                            pcm16k = resample_poly(pcm, up=2, down=3).astype(np.float32, copy=False)
                            await self.ring16k.append(pcm16k)
                            await self.audio_track_buffer.append_from_24k(pcm)
                            self.decoded_seconds += pcm.size / 24000.0
                            self.last_rx_epoch = time.time()
                            self._write_status()
                except Exception as e:
                    self.connected = False
                    self._write_status()
                    print(f"[mirror] connection error: {e!r}")
                    await asyncio.sleep(max(0.2, self.reconnect_delay_seconds))


class PersonaPlexChatBridge:
    def __init__(
        self,
        ws_url: str,
        session: SessionState,
        pcm_ring_24k: PcmRingBuffer,
        pcm_ring_16k: PcmRingBuffer,
        audio_track_buffer: AudioTrackBuffer,
        reconnect_delay_seconds: float,
    ):
        self.ws_url = ws_url
        self.session = session
        self.pcm_ring_24k = pcm_ring_24k
        self.pcm_ring_16k = pcm_ring_16k
        self.audio_track_buffer = audio_track_buffer
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.stop_event = asyncio.Event()
        self.handshake = asyncio.Event()
        self.uplink_queue = asyncio.Queue(maxsize=64)
        self.reader = sphn.OpusStreamReader(24000)
        self.writer = sphn.OpusStreamWriter(24000)
        self.last_error = ""

    async def push_uplink_pcm24k(self, pcm24k: np.ndarray) -> None:
        if pcm24k.size == 0:
            return
        item = pcm24k.astype(np.float32, copy=False).copy()
        if self.uplink_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                _ = self.uplink_queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self.uplink_queue.put_nowait(item)
            self.session.personaplex_audio_frames_tx += 1

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.BINARY:
                continue
            data = msg.data
            if not isinstance(data, (bytes, bytearray)) or len(data) < 2:
                continue
            kind = data[0]
            payload = bytes(data[1:])
            if kind == 0:
                self.handshake.set()
                self.session.personaplex_connected = True
                continue
            if kind != 1:
                continue
            self.reader.append_bytes(payload)
            pcm24k = self.reader.read_pcm()
            if pcm24k.shape[-1] == 0:
                continue
            pcm24k = np.asarray(pcm24k, dtype=np.float32).reshape(-1)
            await self.pcm_ring_24k.append(pcm24k)
            pcm16k = resample_poly(pcm24k, up=2, down=3).astype(np.float32, copy=False)
            await self.pcm_ring_16k.append(pcm16k)
            await self.audio_track_buffer.append_from_24k(pcm24k)
            self.session.personaplex_audio_frames_rx += 1
            self.session.personaplex_last_rx_epoch = time.time()

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.handshake.wait(), timeout=8.0)
        while not self.stop_event.is_set() and not ws.closed:
            try:
                pcm24k = await asyncio.wait_for(self.uplink_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if pcm24k.size == 0:
                continue
            self.writer.append_pcm(pcm24k)
            pages = self.writer.read_bytes()
            if pages:
                await ws.send_bytes(b"\x01" + pages)

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            while not self.stop_event.is_set():
                self.handshake.clear()
                try:
                    async with session.ws_connect(self.ws_url, heartbeat=20.0) as ws:
                        self.session.personaplex_connected = True
                        recv_task = asyncio.create_task(self._recv_loop(ws))
                        send_task = asyncio.create_task(self._send_loop(ws))
                        done, pending = await asyncio.wait(
                            [recv_task, send_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()
                        for task in pending:
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await task
                        for task in done:
                            exc = task.exception()
                            if exc is not None:
                                raise exc
                except Exception as e:
                    self.last_error = repr(e)
                    self.session.personaplex_connected = False
                    if not self.stop_event.is_set():
                        print(f"[personaplex-chat] bridge error: {e!r}")
                if not self.stop_event.is_set():
                    await asyncio.sleep(max(0.2, self.reconnect_delay_seconds))
        self.session.personaplex_connected = False


class MuseTalkRealtimeEngine:
    def __init__(
        self,
        args: AppArgs,
        pcm_ring_16k: PcmRingBuffer,
        video_buffer: VideoFrameBuffer,
    ):
        self.args = args
        self.pcm_ring = pcm_ring_16k
        self.video_buffer = video_buffer
        self.stop_event = asyncio.Event()
        self.last_total_samples = -1
        self.avatar_frame_idx = 0
        self.jobs = 0
        self.dropped_audio_ms_total = 0.0
        self.last_publish_epoch = 0.0
        self.last_error = ""
        self.prev_mouth_patch = None

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
        # Seed a non-black idle frame so preview is visible before first audio packets arrive.
        if getattr(self.avatar, "frame_list_cycle", None):
            try:
                self.video_buffer.last_frame = self.avatar.frame_list_cycle[0].copy()
            except Exception:
                pass

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

    def _infer_window_frames(self, pcm16k_window: np.ndarray, new_frames: int) -> list[np.ndarray]:
        rt = self.rt
        feature_ret = rt.audio_processor.get_audio_feature_from_array(
            pcm16k_window, sample_rate=16000, weight_dtype=rt.weight_dtype
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
                # Temporal smoothing on the mouth patch reduces flicker/chin artifacts.
                if 0.0 <= self.args.mouth_smoothing_alpha < 1.0:
                    x1c = max(0, min(frame.shape[1] - 1, x1))
                    x2c = max(1, min(frame.shape[1], x2))
                    y1c = max(0, min(frame.shape[0] - 1, y1))
                    y2c = max(1, min(frame.shape[0], y2))
                    cur = frame[y1c:y2c, x1c:x2c]
                    if self.prev_mouth_patch is not None and self.prev_mouth_patch.shape == cur.shape:
                        alpha = float(self.args.mouth_smoothing_alpha)
                        cur = cv2.addWeighted(cur, alpha, self.prev_mouth_patch, 1.0 - alpha, 0.0)
                        frame[y1c:y2c, x1c:x2c] = cur
                    self.prev_mouth_patch = frame[y1c:y2c, x1c:x2c].copy()
                combined_frames.append(frame)
                self.avatar_frame_idx += 1

        if not combined_frames:
            return []
        if new_frames <= 0:
            return [combined_frames[-1]]
        return combined_frames[-new_frames:]

    async def run(self) -> None:
        window_samples = int((self.args.window_ms / 1000.0) * 16000)
        min_samples = int((self.args.min_window_ms / 1000.0) * 16000)
        max_advance_samples = int((self.args.max_advance_ms / 1000.0) * 16000)
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

            if max_advance_samples > 0 and new_samples > max_advance_samples:
                dropped = (new_samples - max_advance_samples) * 1000.0 / 16000.0
                self.dropped_audio_ms_total += dropped
                new_samples = max_advance_samples

            # Fixes "same lips over and over" by publishing only newly advanced tail frames.
            new_frames = max(1, int(round((new_samples / 16000.0) * self.args.fps)))
            new_frames = min(new_frames, max(1, self.args.max_tail_frames))
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
            "dropped_audio_ms_total": round(self.dropped_audio_ms_total, 1),
        }


class MuseTalkVideoTrack(VideoStreamTrack):
    def __init__(self, buffer: VideoFrameBuffer, fps: int, stats: dict):
        super().__init__()
        self.buffer = buffer
        self.fps = max(1, fps)
        self.frame_interval = 1.0 / self.fps
        self.stats = stats

    async def recv(self):
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc/av is not installed")
        frame_bgr = await self.buffer.get(timeout=self.frame_interval)
        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        self.stats["video_frames_sent"] = self.stats.get("video_frames_sent", 0) + 1
        self.stats["last_video_send_epoch"] = time.time()
        return video_frame


class MuseTalkAudioTrack(AudioStreamTrack):
    def __init__(self, audio_buffer: AudioTrackBuffer, stats: dict):
        super().__init__()
        self.audio_buffer = audio_buffer
        self.samples_per_frame = 960  # 20ms @ 48kHz
        self.sample_rate = 48000
        self.stats = stats

    async def recv(self):
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc/av is not installed")
        if self.readyState != "live":
            raise MediaStreamError
        pcm = await self.audio_buffer.pop_48k(self.samples_per_frame)
        pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)

        # AudioStreamTrack in aiortc does not provide next_timestamp() (video-only helper),
        # so we keep our own 48k clock here.
        if hasattr(self, "_timestamp"):
            self._timestamp += self.samples_per_frame
            wait = self._start + (self._timestamp / self.sample_rate) - time.time()
            await asyncio.sleep(max(0.0, wait))
        else:
            self._start = time.time()
            self._timestamp = 0

        frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples_per_frame)
        frame.sample_rate = 48000
        frame.planes[0].update(pcm_i16.tobytes())
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self.stats["audio_frames_sent"] = self.stats.get("audio_frames_sent", 0) + 1
        self.stats["last_audio_send_epoch"] = time.time()
        return frame


class WebRtcApp:
    def __init__(self, args: AppArgs):
        self.args = args
        self.started_epoch = time.time()
        self.pcs = set()
        self.pc_states = {}
        self.sessions: dict[str, SessionState] = {}
        self.active_session_id: Optional[str] = None
        self.track_stats = {
            "video_frames_sent": 0,
            "audio_frames_sent": 0,
            "last_video_send_epoch": 0.0,
            "last_audio_send_epoch": 0.0,
        }

        ring_samples = int(args.ring_buffer_seconds * 24000)
        self.pcm_ring_24k = PcmRingBuffer(max_samples=ring_samples)
        self.pcm_ring_16k = PcmRingBuffer(max_samples=int(args.ring_buffer_seconds * 16000))
        self.audio_track_buffer = AudioTrackBuffer(max_samples_48k=int(args.ring_buffer_seconds * 48000))
        self.video_buffer = VideoFrameBuffer(maxsize=args.video_queue_size)

        self.mirror_client: Optional[PersonaPlexMirrorClient] = None
        self.engine: Optional[MuseTalkRealtimeEngine] = None
        if not args.web_test_only:
            if not self._personaplex_chat_enabled():
                ws_url = self._build_mirror_ws_url()
                self.mirror_client = PersonaPlexMirrorClient(
                    ws_url=ws_url,
                    pcm_ring_24k=self.pcm_ring_24k,
                    pcm_ring_16k=self.pcm_ring_16k,
                    audio_track_buffer=self.audio_track_buffer,
                    status_json=args.status_json,
                    reconnect_delay_seconds=args.reconnect_delay_seconds,
                )
            self.engine = MuseTalkRealtimeEngine(
                args=args,
                pcm_ring_16k=self.pcm_ring_16k,
                video_buffer=self.video_buffer,
            )
        else:
            self._seed_web_test_frame()

        self.mirror_task: Optional[asyncio.Task] = None
        self.engine_task: Optional[asyncio.Task] = None
        self.session_cleanup_task: Optional[asyncio.Task] = None

    def _personaplex_chat_enabled(self) -> bool:
        if self.args.web_test_only:
            return False
        return self.args.personaplex_path.rstrip("/").endswith("/api/chat")

    def _seed_web_test_frame(self) -> None:
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = 34
        frame[:, :, 1] = 22
        frame[:, :, 2] = 18
        cv2.rectangle(frame, (40, 40), (w - 40, h - 40), (90, 160, 230), 2)
        cv2.putText(
            frame,
            "MuseTalk WebRTC Test Mode",
            (90, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (240, 240, 240),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "No model loading (connection test only)",
            (90, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        self.video_buffer.last_frame = frame

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    async def _safe_json(self, request: web.Request) -> dict:
        if not request.can_read_body:
            return {}
        with contextlib.suppress(Exception):
            body = await request.json()
            if isinstance(body, dict):
                return body
        return {}

    def _touch_session(self, session: SessionState) -> None:
        session.last_activity_epoch = time.time()

    def _session_state(self, session: SessionState) -> str:
        if session.pcid:
            return self.pc_states.get(session.pcid, "new")
        return "new"

    def _session_should_expire(self, session: SessionState, now: float) -> tuple[bool, str]:
        age = now - session.created_epoch
        if age > max(1.0, self.args.session_max_age_seconds):
            return True, "max_age_exceeded"

        state = self._session_state(session)
        if session.pc is None and age > max(1.0, self.args.session_offer_timeout_seconds):
            return True, "offer_timeout"
        if state in {"closed", "failed", "disconnected"}:
            idle = now - session.last_activity_epoch
            if idle > max(5.0, self.args.session_cleanup_interval_seconds * 2.0):
                return True, f"pc_{state}"
        return False, ""

    async def _close_session(self, session: SessionState, reason: str) -> None:
        sid = session.session_id
        session.close_reason = reason
        if session.personaplex_bridge is not None:
            session.personaplex_bridge.stop_event.set()
        if session.personaplex_bridge_task is not None:
            session.personaplex_bridge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await session.personaplex_bridge_task
            session.personaplex_bridge_task = None
        session.personaplex_bridge = None
        session.personaplex_connected = False
        if session.inbound_task is not None:
            session.inbound_task.cancel()
            session.inbound_task = None
        if session.pc is not None:
            with contextlib.suppress(Exception):
                await session.pc.close()
            self.pcs.discard(session.pc)
            session.pc = None
        if session.pcid:
            self.pc_states[session.pcid] = "closed"
        self.sessions.pop(sid, None)
        if self.active_session_id == sid:
            self.active_session_id = None

    async def _expire_stale_sessions(self) -> None:
        now = time.time()
        expired: list[tuple[SessionState, str]] = []
        for session in list(self.sessions.values()):
            should_expire, reason = self._session_should_expire(session, now)
            if should_expire:
                expired.append((session, reason))
        for session, reason in expired:
            await self._close_session(session, reason)

    async def _session_cleanup_loop(self) -> None:
        interval = max(1.0, float(self.args.session_cleanup_interval_seconds))
        while True:
            await asyncio.sleep(interval)
            with contextlib.suppress(Exception):
                await self._expire_stale_sessions()

    def _aiortc_ice_servers(self):
        servers = []
        for url in self.args.ice_servers:
            if not str(url).strip():
                continue
            with contextlib.suppress(Exception):
                kw = {"urls": url}
                if self.args.ice_username and str(url).lower().startswith(("turn:", "turns:")):
                    kw["username"] = self.args.ice_username
                if self.args.ice_credential and str(url).lower().startswith(("turn:", "turns:")):
                    kw["credential"] = self.args.ice_credential
                servers.append(RTCIceServer(**kw))
        return servers

    def _browser_rtc_config(self) -> dict:
        servers = []
        for url in self.args.ice_servers:
            if not str(url).strip():
                continue
            entry = {"urls": url}
            if self.args.ice_username and str(url).lower().startswith(("turn:", "turns:")):
                entry["username"] = self.args.ice_username
            if self.args.ice_credential and str(url).lower().startswith(("turn:", "turns:")):
                entry["credential"] = self.args.ice_credential
            servers.append(entry)
        return {"iceServers": servers, "iceTransportPolicy": self.args.ice_transport_policy}

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
        if self.mirror_client is not None and self.args.input_source in ("mirror", "mixed"):
            self.mirror_task = asyncio.create_task(self.mirror_client.run())
        if self.engine is not None:
            self.engine_task = asyncio.create_task(self.engine.run())
        self.session_cleanup_task = asyncio.create_task(self._session_cleanup_loop())

    async def on_cleanup(self, _app: web.Application):
        if self.mirror_client is not None:
            self.mirror_client.stop_event.set()
        if self.engine is not None:
            self.engine.stop_event.set()
        for task in (self.mirror_task, self.engine_task, self.session_cleanup_task):
            if task is not None:
                task.cancel()
        for task in (self.mirror_task, self.engine_task, self.session_cleanup_task):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        for session in list(self.sessions.values()):
            with contextlib.suppress(Exception):
                await self._close_session(session, reason="shutdown")
        await asyncio.gather(*(pc.close() for pc in list(self.pcs)), return_exceptions=True)
        self.pcs.clear()
        self.sessions.clear()
        self.active_session_id = None

    async def index(self, _request: web.Request) -> web.Response:
        return web.Response(text=HTML_PAGE, content_type="text/html")

    async def config(self, _request: web.Request) -> web.Response:
        return web.json_response(self._browser_rtc_config())

    async def status(self, _request: web.Request) -> web.Response:
        await self._expire_stale_sessions()
        if self.mirror_client is None:
            mirror_payload = {
                "enabled": False,
                "connected": False,
                "packets": 0,
                "decoded_seconds": 0.0,
                "last_rx_epoch": None,
            }
        else:
            mirror_payload = {
                "enabled": True,
                "connected": self.mirror_client.connected,
                "packets": self.mirror_client.packets,
                "decoded_seconds": self.mirror_client.decoded_seconds,
                "last_rx_epoch": self.mirror_client.last_rx_epoch or None,
            }
        if self.engine is None:
            engine_payload = {
                "mode": "web_test_only",
                "jobs": 0,
                "last_publish_epoch": None,
                "last_error": None,
            }
        else:
            engine_payload = self.engine.status()
        payload = {
            "uptime_seconds": round(time.time() - self.started_epoch, 1),
            "mode": "web_test_only" if self.args.web_test_only else "musetalk",
            "personaplex": {
                "path": self.args.personaplex_path,
                "duplex_chat_mode": self._personaplex_chat_enabled(),
                "text_prompt_set": bool(self.args.personaplex_text_prompt),
                "voice_prompt": self.args.personaplex_voice_prompt or None,
            },
            "mirror": mirror_payload,
            "engine": engine_payload,
            "webrtc": {
                "peer_count": len(self.pcs),
                "peer_states": self.pc_states,
                "session_count": len(self.sessions),
                "active_session_id": self.active_session_id,
                "single_session_mode": self.args.single_session_mode,
                "track_stats": self.track_stats,
                "ice": {
                    "servers": self.args.ice_servers,
                    "transport_policy": self.args.ice_transport_policy,
                },
                "auth_enabled": self.args.enable_api_auth,
            },
        }
        return web.json_response(payload)

    async def snapshot(self, _request: web.Request) -> web.Response:
        data = self.video_buffer.snapshot_jpeg()
        if not data:
            return web.Response(status=503, text="snapshot unavailable")
        return web.Response(body=data, content_type="image/jpeg")

    async def healthz(self, _request: web.Request) -> web.Response:
        await self._expire_stale_sessions()
        return web.json_response(
            {
                "ok": True,
                "aiortc_available": AIORTC_AVAILABLE,
                "uptime_seconds": round(time.time() - self.started_epoch, 1),
                "active_session_id": self.active_session_id,
                "session_count": len(self.sessions),
            }
        )

    async def config_v1(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "rtc_config": self._browser_rtc_config(),
                "single_session_mode": self.args.single_session_mode,
                "session_token_header": SESSION_TOKEN_HEADER,
                "auth_enabled": self.args.enable_api_auth,
            }
        )

    async def active_session(self, _request: web.Request) -> web.Response:
        await self._expire_stale_sessions()
        session = self._active_session()
        if session is None:
            return web.json_response({"active_session": None})
        return web.json_response({"active_session": self._session_payload(session)})

    def _active_session(self) -> Optional[SessionState]:
        if self.active_session_id:
            session = self.sessions.get(self.active_session_id)
            if session is not None:
                return session
            self.active_session_id = None
        for session in self.sessions.values():
            state = self._session_state(session)
            if state not in {"closed", "failed"}:
                self.active_session_id = session.session_id
                return session
        return None

    def _create_session(self) -> SessionState:
        now = time.time()
        session = SessionState(
            session_id=uuid.uuid4().hex,
            token=secrets.token_urlsafe(24),
            created_epoch=now,
            last_activity_epoch=now,
        )
        self.sessions[session.session_id] = session
        self.active_session_id = session.session_id
        return session

    def _get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def _session_conflict_payload(self, session: SessionState) -> dict:
        return {
            "error": "single-session mode already has an active session",
            "active_session": self._session_payload(session),
            "hint": "Pass {\"replace\": true} to POST /v1/sessions to replace it.",
        }

    async def create_session(self, request: web.Request) -> web.Response:
        await self._expire_stale_sessions()
        body = await self._safe_json(request)
        replace = self._coerce_bool(body.get("replace"), default=False) or self._coerce_bool(
            request.query.get("replace"),
            default=False,
        )
        if self.args.single_session_mode:
            active = self._active_session()
            if active is not None:
                if not replace:
                    return web.json_response(self._session_conflict_payload(active), status=409)
                await self._close_session(active, reason="replaced")

        session = self._create_session()
        return web.json_response(
            {
                "session_id": session.session_id,
                "token": session.token,
                "created_epoch": session.created_epoch,
                "session_token_header": SESSION_TOKEN_HEADER,
                "single_session_mode": self.args.single_session_mode,
            }
        )

    def _check_session_token(self, request: web.Request, session: SessionState) -> Optional[web.Response]:
        got = request.headers.get(SESSION_TOKEN_HEADER, "")
        if not got:
            return web.json_response({"error": f"missing {SESSION_TOKEN_HEADER} header"}, status=401)
        if not secrets.compare_digest(got, session.token):
            return web.json_response({"error": "invalid session token"}, status=401)
        return None

    async def delete_session(self, request: web.Request) -> web.Response:
        await self._expire_stale_sessions()
        sid = request.match_info["session_id"]
        session = self._get_session(sid)
        if session is None:
            return web.json_response({"error": "session not found"}, status=404)
        token_err = self._check_session_token(request, session)
        if token_err is not None:
            return token_err
        await self._close_session(session, reason="client_delete")
        return web.json_response({"ok": True, "session_id": sid})

    async def session_stats(self, request: web.Request) -> web.Response:
        await self._expire_stale_sessions()
        sid = request.match_info["session_id"]
        session = self._get_session(sid)
        if session is None:
            return web.json_response({"error": "session not found"}, status=404)
        token_err = self._check_session_token(request, session)
        if token_err is not None:
            return token_err
        self._touch_session(session)
        return web.json_response(self._session_payload(session))

    def _session_payload(self, session: SessionState) -> dict:
        now = time.time()
        return {
            "session_id": session.session_id,
            "created_epoch": session.created_epoch,
            "age_seconds": round(max(0.0, now - session.created_epoch), 3),
            "last_activity_epoch": session.last_activity_epoch,
            "last_offer_epoch": session.last_offer_epoch or None,
            "pc_state": self.pc_states.get(session.pcid or "", "new"),
            "mic_frames_rx": session.mic_frames_rx,
            "mic_samples_rx_16k": session.mic_samples_rx_16k,
            "last_mic_rx_epoch": session.last_mic_rx_epoch or None,
            "personaplex_connected": session.personaplex_connected,
            "personaplex_audio_frames_tx": session.personaplex_audio_frames_tx,
            "personaplex_audio_frames_rx": session.personaplex_audio_frames_rx,
            "personaplex_last_rx_epoch": session.personaplex_last_rx_epoch or None,
            "active": session.session_id == self.active_session_id,
            "close_reason": session.close_reason or None,
        }

    async def _start_personaplex_chat_bridge(self, session: SessionState) -> None:
        if not self._personaplex_chat_enabled():
            return
        if session.personaplex_bridge is not None:
            session.personaplex_bridge.stop_event.set()
        if session.personaplex_bridge_task is not None:
            session.personaplex_bridge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await session.personaplex_bridge_task
            session.personaplex_bridge_task = None

        ws_url = self._build_mirror_ws_url()
        bridge = PersonaPlexChatBridge(
            ws_url=ws_url,
            session=session,
            pcm_ring_24k=self.pcm_ring_24k,
            pcm_ring_16k=self.pcm_ring_16k,
            audio_track_buffer=self.audio_track_buffer,
            reconnect_delay_seconds=self.args.reconnect_delay_seconds,
        )
        session.personaplex_bridge = bridge
        session.personaplex_bridge_task = asyncio.create_task(bridge.run())
        print(f"[personaplex-chat] started for session={session.session_id} url={ws_url}")

    async def offer(self, request: web.Request) -> web.Response:
        await self._expire_stale_sessions()
        if self.args.single_session_mode:
            active = self._active_session()
            if active is not None:
                await self._close_session(active, reason="legacy_offer_replaced")
        session = self._create_session()
        return await self._handle_offer(request, session=session, include_session_token=True)

    async def session_offer(self, request: web.Request) -> web.Response:
        await self._expire_stale_sessions()
        sid = request.match_info["session_id"]
        session = self._get_session(sid)
        if session is None:
            return web.json_response({"error": "session not found"}, status=404)
        token_err = self._check_session_token(request, session)
        if token_err is not None:
            return token_err
        if self.args.single_session_mode and self.active_session_id and self.active_session_id != sid:
            active = self._active_session()
            if active is not None:
                return web.json_response(self._session_conflict_payload(active), status=409)
        return await self._handle_offer(request, session=session)

    async def _handle_offer(
        self,
        request: web.Request,
        session: SessionState,
        include_session_token: bool = False,
    ) -> web.Response:
        if not AIORTC_AVAILABLE:
            return web.json_response(
                {"error": "aiortc/av not installed. Install: pip install aiortc av"},
                status=500,
            )
        params = await self._safe_json(request)
        sdp = params.get("sdp")
        offer_type = str(params.get("type", "")).lower()
        if not isinstance(sdp, str) or not sdp.strip() or offer_type != "offer":
            return web.json_response({"error": "invalid offer payload; expected {type:'offer', sdp:'...'}"}, status=400)
        offer = RTCSessionDescription(sdp=sdp, type=offer_type)
        if session.pc is not None:
            with contextlib.suppress(Exception):
                await session.pc.close()
            self.pcs.discard(session.pc)

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=self._aiortc_ice_servers()))
        self.pcs.add(pc)
        pcid = f"pc_{id(pc)}"
        self.pc_states[pcid] = pc.connectionState
        session.pc = pc
        session.pcid = pcid
        session.last_offer_epoch = time.time()
        self._touch_session(session)
        self.active_session_id = session.session_id
        await self._start_personaplex_chat_bridge(session)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.pc_states[pcid] = pc.connectionState
            self._touch_session(session)
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self.pcs.discard(pc)
                self.pc_states[pcid] = "closed"
                if session.inbound_task is not None:
                    session.inbound_task.cancel()
                    session.inbound_task = None

        @pc.on("track")
        async def on_track(track):
            if getattr(track, "kind", "") != "audio":
                return
            if session.inbound_task is not None:
                session.inbound_task.cancel()
            session.inbound_task = asyncio.create_task(self._consume_inbound_audio(track, session))
            self._touch_session(session)

        video_track = MuseTalkVideoTrack(self.video_buffer, fps=self.args.fps, stats=self.track_stats)
        audio_track = MuseTalkAudioTrack(self.audio_track_buffer, stats=self.track_stats)
        pc.addTrack(video_track)
        pc.addTrack(audio_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self._wait_for_ice_gathering(pc, timeout=3.0)

        payload = {
            "session_id": session.session_id,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
        if include_session_token:
            payload["session_token"] = session.token
            payload["session_token_header"] = SESSION_TOKEN_HEADER
        return web.json_response(payload)

    async def _consume_inbound_audio(self, track, session: SessionState) -> None:
        while True:
            frame = await track.recv()
            pcm = frame.to_ndarray()
            pcm = np.asarray(pcm)
            if pcm.ndim == 2:
                pcm = pcm.mean(axis=0)
            pcm = pcm.reshape(-1).astype(np.float32, copy=False)
            fmt_name = str(getattr(getattr(frame, "format", None), "name", "")).lower()
            if fmt_name.startswith(("s16", "s32", "u8")):
                pcm = pcm / 32768.0
            src_sr = int(getattr(frame, "sample_rate", 48000) or 48000)
            if pcm.size == 0:
                continue

            pcm16k = self._resample_audio(pcm, src_sr, 16000)
            session.mic_samples_rx_16k += int(pcm16k.size)
            session.mic_frames_rx += 1
            session.last_mic_rx_epoch = time.time()
            self._touch_session(session)

            if session.personaplex_bridge is not None:
                pcm24k_uplink = self._resample_audio(pcm, src_sr, 24000)
                await session.personaplex_bridge.push_uplink_pcm24k(pcm24k_uplink)

            if self.args.input_source in ("webrtc", "mixed") and session.personaplex_bridge is None:
                await self.pcm_ring_16k.append(pcm16k)

            if self.args.webrtc_audio_loopback and session.personaplex_bridge is None:
                pcm24k = self._resample_audio(pcm, src_sr, 24000)
                await self.pcm_ring_24k.append(pcm24k)
                await self.audio_track_buffer.append_from_24k(pcm24k)

    def _resample_audio(self, pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return pcm.astype(np.float32, copy=False)
        if src_sr <= 0 or dst_sr <= 0:
            return np.zeros((0,), dtype=np.float32)
        return resample_poly(pcm, up=dst_sr, down=src_sr).astype(np.float32, copy=False)

    @web.middleware
    async def auth_middleware(self, request: web.Request, handler):
        if not self.args.enable_api_auth:
            return await handler(request)
        if not request.path.startswith("/v1/"):
            return await handler(request)
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {self.args.api_token}"
        if not self.args.api_token or auth != expected:
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    async def _wait_for_ice_gathering(self, pc: RTCPeerConnection, timeout: float) -> None:
        if pc.iceGatheringState == "complete":
            return
        done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        async def _on_ice_state_change():
            if pc.iceGatheringState == "complete":
                done.set()

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(done.wait(), timeout=timeout)

    def build_app(self) -> web.Application:
        app = web.Application(middlewares=[self.auth_middleware])
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        app.router.add_get("/", self.index)
        app.router.add_get("/healthz", self.healthz)
        app.router.add_get("/config", self.config)
        app.router.add_get("/status", self.status)
        app.router.add_get("/snapshot.jpg", self.snapshot)
        app.router.add_post("/offer", self.offer)
        app.router.add_get("/v1/config", self.config_v1)
        app.router.add_get("/v1/session", self.active_session)
        app.router.add_post("/v1/sessions", self.create_session)
        app.router.add_post("/v1/sessions/{session_id}/offer", self.session_offer)
        app.router.add_get("/v1/sessions/{session_id}/stats", self.session_stats)
        app.router.add_delete("/v1/sessions/{session_id}", self.delete_session)
        return app


def parse_args() -> AppArgs:
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
        enable_api_auth=ns.enable_api_auth,
        api_token=ns.api_token,
        session_offer_timeout_seconds=ns.session_offer_timeout_seconds,
        session_max_age_seconds=ns.session_max_age_seconds,
        session_cleanup_interval_seconds=ns.session_cleanup_interval_seconds,
        single_session_mode=(not ns.multi_session),
        web_test_only=ns.web_test_only,
    )


def main():
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
    print(f"[webrtc] single_session_mode={args.single_session_mode} session_token_header={SESSION_TOKEN_HEADER}")
    print(f"[webrtc] personaplex_path={args.personaplex_path} chat_mode={chat_mode}")
    if args.web_test_only:
        print("[webrtc] web-test-only enabled (MuseTalk model initialization skipped)")
    if args.enable_api_auth:
        print("[webrtc] /v1 auth middleware enabled")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
