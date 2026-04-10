"""In-memory audio/video buffering primitives."""

import asyncio
import contextlib

import cv2
import numpy as np

class PcmRingBuffer:
    """Thread-safe asyncio ring buffer for mono float PCM samples."""

    def __init__(self, max_samples: int):
        """Create a lock-protected mono PCM ring buffer.

        Receives:
        - `max_samples`: maximum retained sample count.

        Returns:
        - `None`.
        """

        self.max_samples = max_samples
        self.buf = np.zeros((0,), dtype=np.float32)
        self.total_samples = 0
        self.lock = asyncio.Lock()

    async def append(self, samples: np.ndarray) -> None:
        """Append float PCM samples and clip to ring capacity.

        Receives:
        - `samples`: mono audio samples (any numeric dtype).

        Returns:
        - `None`.
        """

        if samples.size == 0:
            return
        async with self.lock:
            self.total_samples += int(samples.size)
            self.buf = np.concatenate([self.buf, samples.astype(np.float32, copy=False)])
            if self.buf.size > self.max_samples:
                self.buf = self.buf[-self.max_samples :]

    async def latest(self, n_samples: int) -> tuple[np.ndarray, int]:
        """Return the newest window and cumulative sample counter.

        Receives:
        - `n_samples`: desired window length from tail of ring.

        Returns:
        - `(window, total_samples_seen)`.
        """

        async with self.lock:
            if self.buf.size <= n_samples:
                return self.buf.copy(), self.total_samples
            return self.buf[-n_samples:].copy(), self.total_samples


class AudioTrackBuffer:
    """Buffered 48k mono sample queue feeding outbound WebRTC audio track."""

    def __init__(self, max_samples_48k: int):
        """Create outbound audio buffer used by WebRTC audio track.

        Receives:
        - `max_samples_48k`: cap in 48kHz mono samples.

        Returns:
        - `None`.
        """

        self.max_samples = max_samples_48k
        self.buf = np.zeros((0,), dtype=np.float32)
        self.lock = asyncio.Lock()

    async def append_from_24k(self, mono24k: np.ndarray) -> None:
        """Append 24kHz mono PCM by upsampling to 48kHz.

        Receives:
        - `mono24k`: float PCM at 24kHz.

        Returns:
        - `None`.
        """

        if mono24k.size == 0:
            return
        mono48k = np.repeat(mono24k.astype(np.float32, copy=False), 2)
        async with self.lock:
            self.buf = np.concatenate([self.buf, mono48k])
            if self.buf.size > self.max_samples:
                self.buf = self.buf[-self.max_samples :]

    async def pop_48k(self, n_samples: int) -> np.ndarray:
        """Pop exactly N 48kHz samples, zero-padding if underflow.

        Receives:
        - `n_samples`: requested frame size.

        Returns:
        - `np.ndarray` sized exactly `n_samples`.
        """

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
    """Bounded queue of BGR frames with last-frame fallback semantics."""

    def __init__(self, maxsize: int):
        """Create bounded queue for generated avatar frames.

        Receives:
        - `maxsize`: queue capacity before dropping oldest frames.

        Returns:
        - `None`.
        """

        self.queue = asyncio.Queue(maxsize=maxsize)
        self.last_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    async def publish(self, frame_bgr: np.ndarray) -> None:
        """Publish a BGR frame, dropping oldest when queue is full.

        Receives:
        - `frame_bgr`: latest video frame.

        Returns:
        - `None`.
        """

        self.last_frame = frame_bgr
        if self.queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                _ = self.queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(frame_bgr)

    async def get(self, timeout: float = 0.12) -> np.ndarray:
        """Return next frame or fallback to last frame on timeout.

        Receives:
        - `timeout`: maximum wait in seconds.

        Returns:
        - BGR frame (`np.ndarray`).
        """

        try:
            frame = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            self.last_frame = frame
            return frame
        except asyncio.TimeoutError:
            return self.last_frame

    def snapshot_jpeg(self) -> bytes:
        """Encode the last known frame to JPEG bytes.

        Receives:
        - None.

        Returns:
        - Encoded JPEG bytes, or empty bytes on encoding failure.
        """

        ok, enc = cv2.imencode(".jpg", self.last_frame)
        if not ok:
            return b""
        return enc.tobytes()

