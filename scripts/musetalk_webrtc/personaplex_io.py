"""PersonaPlex websocket clients: mirror mode and duplex chat bridge."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
import numpy as np
import sphn
from scipy.signal import resample_poly

from .buffers import AudioTrackBuffer, PcmRingBuffer
from .models import SessionState

class PersonaPlexMirrorClient:
    """Read-only websocket client for PersonaPlex avatar audio mirror stream."""

    def __init__(
        self,
        ws_url: str,
        pcm_ring_24k: PcmRingBuffer,
        pcm_ring_16k: PcmRingBuffer,
        audio_track_buffer: AudioTrackBuffer,
        status_json: Optional[Path],
        reconnect_delay_seconds: float,
    ):
        """Create read-only PersonaPlex mirror websocket client.

        Receives:
        - `ws_url`: mirror websocket endpoint (usually `/api/avatar/audio`).
        - Output buffers for 24k ring, 16k ring, and outbound WebRTC audio.
        - Optional `status_json` file path for live diagnostics.
        - Reconnect delay.

        Returns:
        - `None`.
        """

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
        """Persist mirror connection metrics to JSON when enabled.

        Receives:
        - None.

        Returns:
        - `None`.
        """

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
        """Run reconnecting websocket receive loop for mirror audio.

        Receives:
        - None.

        Returns:
        - `None` (runs until `stop_event` is set).
        """

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
    """Per-session duplex websocket bridge between browser mic and PersonaPlex."""

    def __init__(
        self,
        ws_url: str,
        session: SessionState,
        pcm_ring_24k: PcmRingBuffer,
        pcm_ring_16k: PcmRingBuffer,
        audio_track_buffer: AudioTrackBuffer,
        reconnect_delay_seconds: float,
        debug_log: Optional[Callable[..., Any]] = None,
    ):
        """Create duplex websocket bridge for a single session (`/api/chat`).

        Receives:
        - `ws_url`: chat websocket endpoint with prompt query params.
        - `session`: owning `SessionState` to update per-session stats.
        - Output audio buffers/rings for inference and outbound playback.
        - Reconnect delay.

        Returns:
        - `None`.
        """

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
        self.handshake_epoch = 0.0
        self.rx_packets = 0
        self.tx_packets = 0
        self.debug_log = debug_log

    async def push_uplink_pcm24k(self, pcm24k: np.ndarray) -> None:
        """Queue browser mic PCM (24k) for upstream websocket send.

        Receives:
        - `pcm24k`: mono float PCM at 24kHz.

        Returns:
        - `None`.
        """

        if pcm24k.size == 0:
            return
        item = pcm24k.astype(np.float32, copy=False).copy()
        if self.uplink_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                _ = self.uplink_queue.get_nowait()
            if self.debug_log is not None:
                self.debug_log(
                    "personaplex_chat.uplink_drop",
                    session_id=self.session.session_id,
                    queue_size=int(self.uplink_queue.qsize()),
                )
        with contextlib.suppress(asyncio.QueueFull):
            self.uplink_queue.put_nowait(item)
            self.session.personaplex_audio_frames_tx += 1

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Receive/decode PersonaPlex audio and fan out to local buffers.

        Receives:
        - Open websocket object.

        Returns:
        - `None` (loop exits when websocket closes/errors).
        """

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
                self.handshake_epoch = time.time()
                self.session.personaplex_connected = True
                if self.debug_log is not None:
                    self.debug_log(
                        "personaplex_chat.handshake",
                        session_id=self.session.session_id,
                        ws_url=self.ws_url,
                    )
                continue
            if kind != 1:
                continue
            self.rx_packets += 1
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
            if self.debug_log is not None and self.rx_packets in {1, 10}:
                self.debug_log(
                    "personaplex_chat.rx_audio",
                    session_id=self.session.session_id,
                    packets=self.rx_packets,
                    pcm_samples=int(pcm24k.size),
                )

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Encode queued uplink PCM to Opus and send binary chat frames.

        Receives:
        - Open websocket object.

        Returns:
        - `None` (loop exits on stop/close).
        """

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
                self.tx_packets += 1
                if self.debug_log is not None and self.tx_packets in {1, 10}:
                    self.debug_log(
                        "personaplex_chat.tx_audio",
                        session_id=self.session.session_id,
                        packets=self.tx_packets,
                        bytes=int(len(pages)),
                    )

    async def run(self) -> None:
        """Run reconnecting duplex bridge using concurrent recv/send tasks.

        Receives:
        - None.

        Returns:
        - `None` (runs until `stop_event` is set).
        """

        async with aiohttp.ClientSession() as session:
            while not self.stop_event.is_set():
                self.handshake.clear()
                try:
                    async with session.ws_connect(self.ws_url, heartbeat=20.0) as ws:
                        self.session.personaplex_connected = True
                        if self.debug_log is not None:
                            self.debug_log(
                                "personaplex_chat.ws_connected",
                                session_id=self.session.session_id,
                                ws_url=self.ws_url,
                            )
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
                    if self.debug_log is not None:
                        self.debug_log(
                            "personaplex_chat.error",
                            session_id=self.session.session_id,
                            error=self.last_error,
                        )
                    if not self.stop_event.is_set():
                        print(f"[personaplex-chat] bridge error: {e!r}")
                if not self.stop_event.is_set():
                    await asyncio.sleep(max(0.2, self.reconnect_delay_seconds))
        self.session.personaplex_connected = False
