import argparse
import asyncio
import contextlib
import json
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiohttp
import numpy as np
import sphn


def decode_audio_to_f32_mono(audio_path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    return pcm


def write_wav_mono_int16(path: Path, pcm_f32: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(pcm_f32, -1.0, 1.0)
    pcm_i16 = (clipped * 32767.0).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_i16.tobytes())


def atomic_write_wav(path: Path, pcm_f32: np.ndarray, sample_rate: int) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_wav_mono_int16(tmp_path, pcm_f32, sample_rate)
    tmp_path.replace(path)


def build_ws_url(
    ws_url: Optional[str],
    host: str,
    port: int,
    path: str,
    text_prompt: str,
    voice_prompt: str,
    extra_query: list[str],
) -> str:
    if ws_url:
        parsed = urlparse(ws_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    else:
        normalized_path = path if path.startswith("/") else f"/{path}"
        parsed = urlparse(f"ws://{host}:{port}{normalized_path}")
        query = {}

    is_chat_path = parsed.path.rstrip("/").endswith("/api/chat")
    if is_chat_path:
        query.setdefault("text_prompt", text_prompt)
        query.setdefault("voice_prompt", voice_prompt)
    else:
        if text_prompt:
            query["text_prompt"] = text_prompt
        if voice_prompt:
            query["voice_prompt"] = voice_prompt
    for kv in extra_query:
        if "=" not in kv:
            raise ValueError(f"Invalid --extra-query '{kv}'. Expected key=value.")
        k, v = kv.split("=", 1)
        query[k] = v
    return urlunparse(parsed._replace(query=urlencode(query)))


@dataclass
class BridgeArgs:
    ws_url: str
    sample_rate: int
    input_wav: Optional[Path]
    input_chunk_samples: int
    send_speed: float
    run_seconds: Optional[float]
    receive_tail_seconds: float
    chunk_path: Path
    chunk_seconds: float
    chunk_hop_ms: int
    min_chunk_seconds: float
    max_buffer_seconds: float
    emit_chunks_dir: Optional[Path]
    reply_wav: Optional[Path]
    verbose_text: bool
    heartbeat_seconds: float
    reconnect_delay_seconds: float
    max_reconnect_attempts: int
    idle_log_seconds: float
    status_json: Optional[Path]


class RollingAudioBuffer:
    def __init__(self, max_samples: int) -> None:
        self._max_samples = max_samples
        self._buf = np.zeros((0,), dtype=np.float32)
        self._total_samples = 0
        self._lock = asyncio.Lock()

    async def append(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        async with self._lock:
            self._total_samples += int(chunk.size)
            self._buf = np.concatenate([self._buf, chunk.astype(np.float32, copy=False)])
            if self._buf.size > self._max_samples:
                self._buf = self._buf[-self._max_samples :]

    async def latest(self, n_samples: int) -> tuple[np.ndarray, int]:
        async with self._lock:
            if self._buf.size <= n_samples:
                return self._buf.copy(), self._total_samples
            return self._buf[-n_samples:].copy(), self._total_samples


class PersonaPlexBridge:
    def __init__(self, args: BridgeArgs) -> None:
        self.args = args
        max_samples = int(args.sample_rate * args.max_buffer_seconds)
        self.buffer = RollingAudioBuffer(max_samples=max_samples)
        self.handshake = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.reader = sphn.OpusStreamReader(args.sample_rate)
        self.reply_chunks: list[np.ndarray] = []
        self.last_chunk_total = -1
        self.chunk_index = 0
        self.audio_packets = 0
        self.audio_samples = 0
        self.last_stats_t = time.time()
        self.last_audio_rx_t = 0.0

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
                        continue
                    kind = data[0]
                    payload = bytes(data[1:])
                    if kind == 0:
                        self.handshake.set()
                        print("[bridge] handshake received")
                    elif kind == 1:
                        self.audio_packets += 1
                        self.reader.append_bytes(payload)
                        pcm = self.reader.read_pcm()
                        if pcm.shape[-1] == 0:
                            continue
                        pcm_f32 = np.asarray(pcm, dtype=np.float32).reshape(-1)
                        self.audio_samples += int(pcm_f32.size)
                        self.last_audio_rx_t = time.time()
                        await self.buffer.append(pcm_f32)
                        self.reply_chunks.append(pcm_f32.copy())
                        now = time.time()
                        if now - self.last_stats_t >= 2.0:
                            sec = self.audio_samples / float(self.args.sample_rate)
                            print(
                                f"[bridge] rx audio packets={self.audio_packets} "
                                f"decoded_sec={sec:.2f}"
                            )
                            self.last_stats_t = now
                    elif kind == 2:
                        if self.args.verbose_text:
                            text = payload.decode("utf-8", errors="replace")
                            print(f"[personaplex-text] {text}", flush=True)
                    else:
                        print(f"[bridge] ignoring ws message kind={kind}, bytes={len(payload)}")
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    print("[bridge] websocket closed by server")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    exc = ws.exception()
                    if exc is None:
                        print("[bridge] websocket reported ERROR with no exception; closing recv loop.")
                    else:
                        print(f"[bridge] websocket error: {exc!r}")
                    break
        finally:
            self.stop_event.set()

    async def _emit_chunk_loop(self) -> None:
        n_window = int(self.args.chunk_seconds * self.args.sample_rate)
        n_min = int(self.args.min_chunk_seconds * self.args.sample_rate)
        hop_sec = self.args.chunk_hop_ms / 1000.0
        while not self.stop_event.is_set():
            await asyncio.sleep(hop_sec)
            chunk, total_samples = await self.buffer.latest(n_window)
            if total_samples == self.last_chunk_total:
                continue
            if chunk.size < n_min:
                continue
            atomic_write_wav(self.args.chunk_path, chunk, self.args.sample_rate)
            if self.chunk_index % 5 == 0:
                print(
                    f"[bridge] emitted rolling chunk: {self.args.chunk_path} "
                    f"(len={chunk.size / self.args.sample_rate:.2f}s)"
                )
            if self.args.emit_chunks_dir is not None:
                self.args.emit_chunks_dir.mkdir(parents=True, exist_ok=True)
                stamped = self.args.emit_chunks_dir / f"chunk_{self.chunk_index:06d}.wav"
                write_wav_mono_int16(stamped, chunk, self.args.sample_rate)
            self.chunk_index += 1
            self.last_chunk_total = total_samples

    async def _send_input_wav(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        if self.args.input_wav is None:
            return
        print(f"[bridge] loading source audio: {self.args.input_wav}")
        pcm = decode_audio_to_f32_mono(self.args.input_wav, self.args.sample_rate)
        if pcm.size == 0:
            raise RuntimeError("input audio decoded to zero samples")

        await asyncio.wait_for(self.handshake.wait(), timeout=20.0)
        writer = sphn.OpusStreamWriter(self.args.sample_rate)
        n = self.args.input_chunk_samples
        print(f"[bridge] streaming {pcm.size / self.args.sample_rate:.2f}s audio to PersonaPlex")
        t0 = time.perf_counter()
        for start in range(0, pcm.size, n):
            chunk = pcm[start : start + n]
            writer.append_pcm(chunk)
            pages = writer.read_bytes()
            if pages:
                await ws.send_bytes(b"\x01" + pages)
            if self.args.send_speed > 0:
                await asyncio.sleep((chunk.size / self.args.sample_rate) / self.args.send_speed)
        pages = writer.read_bytes()
        if pages:
            await ws.send_bytes(b"\x01" + pages)
        print(f"[bridge] finished streaming input in {time.perf_counter() - t0:.2f}s")

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            attempts = 0
            last_idle_log_t = 0.0
            while True:
                self.stop_event.clear()
                self.handshake.clear()
                recv_task = None
                emit_task = None
                send_task = None
                try:
                    hb = self.args.heartbeat_seconds if self.args.heartbeat_seconds > 0 else None
                    print(f"[bridge] connecting: {self.args.ws_url}")
                    async with session.ws_connect(self.args.ws_url, heartbeat=hb) as ws:
                        recv_task = asyncio.create_task(self._recv_loop(ws))
                        emit_task = asyncio.create_task(self._emit_chunk_loop())
                        send_task = asyncio.create_task(self._send_input_wav(ws))

                        if self.args.input_wav is not None:
                            await send_task
                            await asyncio.sleep(self.args.receive_tail_seconds)
                            self.stop_event.set()
                        elif self.args.run_seconds is not None:
                            await asyncio.sleep(self.args.run_seconds)
                            self.stop_event.set()
                        else:
                            while not self.stop_event.is_set():
                                await asyncio.sleep(0.5)
                                now = time.time()
                                if now - last_idle_log_t >= max(1.0, self.args.idle_log_seconds):
                                    if self.last_audio_rx_t == 0.0:
                                        print("[bridge][debug] connected, waiting for assistant audio from PersonaPlex...")
                                    else:
                                        delta = now - self.last_audio_rx_t
                                        print(f"[bridge][debug] connected, last audio {delta:.1f}s ago")
                                    last_idle_log_t = now
                                self._write_status()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    attempts += 1
                    print(f"[bridge] connection error: {e!r}")
                finally:
                    self.stop_event.set()
                    for task in (send_task, emit_task, recv_task):
                        if task is None:
                            continue
                        task.cancel()
                    for task in (send_task, emit_task, recv_task):
                        if task is None:
                            continue
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await task
                    self._write_status()

                if self.args.input_wav is not None or self.args.run_seconds is not None:
                    break
                if self.args.max_reconnect_attempts > 0 and attempts >= self.args.max_reconnect_attempts:
                    print("[bridge] max reconnect attempts reached, exiting.")
                    break
                await asyncio.sleep(max(0.1, self.args.reconnect_delay_seconds))

        if self.args.reply_wav is not None and self.reply_chunks:
            full_reply = np.concatenate(self.reply_chunks)
            write_wav_mono_int16(self.args.reply_wav, full_reply, self.args.sample_rate)
            print(f"[bridge] wrote full reply wav: {self.args.reply_wav}")
        print(f"[bridge] latest rolling chunk: {self.args.chunk_path}")

    def _write_status(self) -> None:
        if self.args.status_json is None:
            return
        payload = {
            "ws_url": self.args.ws_url,
            "chunk_path": str(self.args.chunk_path),
            "audio_packets": self.audio_packets,
            "decoded_seconds": self.audio_samples / float(self.args.sample_rate),
            "chunk_index": self.chunk_index,
            "last_audio_rx_epoch": self.last_audio_rx_t or None,
        }
        tmp = self.args.status_json.with_suffix(".tmp.json")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.args.status_json)


def parse_args() -> BridgeArgs:
    parser = argparse.ArgumentParser(
        description="PersonaPlex -> rolling WAV bridge for MuseTalk realtime chunking."
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default=None,
        help="Full websocket URL. If omitted, built from host/port/query flags.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="PersonaPlex host")
    parser.add_argument("--port", type=int, default=8998, help="PersonaPlex port")
    parser.add_argument(
        "--path",
        type=str,
        default="/api/avatar/audio",
        help="Websocket path when --ws-url is not provided. Use /api/chat to run active query-based chat sessions.",
    )
    parser.add_argument("--text-prompt", type=str, default="", help="PersonaPlex text_prompt query value")
    parser.add_argument(
        "--voice-prompt",
        type=str,
        default="",
        help="PersonaPlex voice_prompt query value (usually required when voice_prompt_dir is enabled).",
    )
    parser.add_argument(
        "--extra-query",
        action="append",
        default=[],
        help="Additional ws query parameters in key=value format. Can be repeated.",
    )
    parser.add_argument("--sample-rate", type=int, default=24000, help="Bridge audio sample rate")
    parser.add_argument("--input-wav", type=str, default=None, help="Optional wav/audio file to send to PersonaPlex")
    parser.add_argument(
        "--input-chunk-samples",
        type=int,
        default=960,
        help="Samples per outgoing encoded chunk to PersonaPlex.",
    )
    parser.add_argument(
        "--send-speed",
        type=float,
        default=1.0,
        help="1.0 for realtime sending, 2.0 for 2x, <=0 for no sleep.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Without --input-wav, keep bridge open for this many seconds then exit.",
    )
    parser.add_argument(
        "--receive-tail-seconds",
        type=float,
        default=2.0,
        help="After sending input wav, extra time to keep receiving response audio.",
    )
    parser.add_argument(
        "--chunk-path",
        type=str,
        default="data/audio/live/latest_chunk.wav",
        help="Rolling chunk path for MuseTalk worker.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.8,
        help="Length of rolling chunk wav window.",
    )
    parser.add_argument(
        "--chunk-hop-ms",
        type=int,
        default=200,
        help="How often to refresh rolling wav chunk.",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=0.4,
        help="Do not emit chunk until this many seconds are available.",
    )
    parser.add_argument(
        "--max-buffer-seconds",
        type=float,
        default=20.0,
        help="In-memory receive ring buffer size.",
    )
    parser.add_argument(
        "--emit-chunks-dir",
        type=str,
        default=None,
        help="Optional directory to persist every emitted chunk for debugging.",
    )
    parser.add_argument(
        "--reply-wav",
        type=str,
        default="data/audio/live/personaplex_reply.wav",
        help="Where to save full received PersonaPlex audio reply.",
    )
    parser.add_argument("--verbose-text", action="store_true", help="Print text tokens received (kind=0x02).")
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=0.0,
        help="aiohttp websocket heartbeat interval; 0 disables heartbeat (recommended for idle mirror mode).",
    )
    parser.add_argument(
        "--reconnect-delay-seconds",
        type=float,
        default=1.5,
        help="Delay before reconnect after socket errors.",
    )
    parser.add_argument(
        "--max-reconnect-attempts",
        type=int,
        default=0,
        help="0 means infinite reconnect attempts for live mirror mode.",
    )
    parser.add_argument(
        "--idle-log-seconds",
        type=float,
        default=5.0,
        help="How often to print waiting logs when connected but no audio packets arrive.",
    )
    parser.add_argument(
        "--status-json",
        type=str,
        default="data/audio/live/bridge_status.json",
        help="Write bridge stats JSON for debugging; empty string disables.",
    )
    ns = parser.parse_args()

    ws_url = build_ws_url(
        ws_url=ns.ws_url,
        host=ns.host,
        port=ns.port,
        path=ns.path,
        text_prompt=ns.text_prompt,
        voice_prompt=ns.voice_prompt,
        extra_query=ns.extra_query,
    )
    is_chat_target = urlparse(ws_url).path.rstrip("/").endswith("/api/chat")
    if is_chat_target and ns.voice_prompt == "":
        print(
            "[bridge][warn] voice_prompt is empty. If your server enforces voice prompts, "
            "set --voice-prompt <filename>."
        )
    return BridgeArgs(
        ws_url=ws_url,
        sample_rate=ns.sample_rate,
        input_wav=Path(ns.input_wav).expanduser().resolve() if ns.input_wav else None,
        input_chunk_samples=ns.input_chunk_samples,
        send_speed=ns.send_speed,
        run_seconds=ns.run_seconds,
        receive_tail_seconds=ns.receive_tail_seconds,
        chunk_path=Path(ns.chunk_path).expanduser().resolve(),
        chunk_seconds=ns.chunk_seconds,
        chunk_hop_ms=ns.chunk_hop_ms,
        min_chunk_seconds=ns.min_chunk_seconds,
        max_buffer_seconds=ns.max_buffer_seconds,
        emit_chunks_dir=Path(ns.emit_chunks_dir).expanduser().resolve() if ns.emit_chunks_dir else None,
        reply_wav=Path(ns.reply_wav).expanduser().resolve() if ns.reply_wav else None,
        verbose_text=ns.verbose_text,
        heartbeat_seconds=ns.heartbeat_seconds,
        reconnect_delay_seconds=ns.reconnect_delay_seconds,
        max_reconnect_attempts=ns.max_reconnect_attempts,
        idle_log_seconds=ns.idle_log_seconds,
        status_json=Path(ns.status_json).expanduser().resolve() if str(ns.status_json).strip() else None,
    )


async def _amain() -> None:
    args = parse_args()
    bridge = PersonaPlexBridge(args)
    await bridge.run()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\n[bridge] stopped by user")


if __name__ == "__main__":
    main()
