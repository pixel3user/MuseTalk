"""In-memory audio/video buffering primitives."""

import asyncio
import contextlib

import cv2
import numpy as np

class PcmRingBuffer:
    """Thread-safe asyncio ring buffer for mono float PCM samples.

    Uses a pre-allocated fixed-size numpy array with wrap-around write index.
    This eliminates the np.concatenate allocation churn that caused GC pressure
    and unpredictable latency spikes during long calls (the old implementation
    reallocated the entire buffer ~50 times/second on 20ms audio chunks).
    """

    def __init__(self, max_samples: int):
        """Create a lock-protected mono PCM ring buffer.

        Receives:
        - `max_samples`: maximum retained sample count.

        Returns:
        - `None`.
        """

        self.max_samples = max_samples
        self._ring = np.zeros(max_samples, dtype=np.float32)
        self._write_pos = 0  # next write index (wraps around)
        self._fill = 0  # how many valid samples are in the ring (0..max_samples)
        self.total_samples = 0
        self.lock = asyncio.Lock()
        self.new_data_event = asyncio.Event()

    async def append(self, samples: np.ndarray) -> None:
        """Append float PCM samples into the circular buffer.

        Receives:
        - `samples`: mono audio samples (any numeric dtype).

        Returns:
        - `None`.
        """

        if samples.size == 0:
            return
        async with self.lock:
            data = samples.astype(np.float32, copy=False).ravel()
            n = data.size
            self.total_samples += n

            if n >= self.max_samples:
                # Input larger than ring — just keep the last max_samples
                self._ring[:] = data[-self.max_samples:]
                self._write_pos = 0
                self._fill = self.max_samples
            else:
                # Write with wrap-around
                end = self._write_pos + n
                if end <= self.max_samples:
                    self._ring[self._write_pos:end] = data
                else:
                    first = self.max_samples - self._write_pos
                    self._ring[self._write_pos:] = data[:first]
                    self._ring[:n - first] = data[first:]
                self._write_pos = end % self.max_samples
                self._fill = min(self._fill + n, self.max_samples)

            self.new_data_event.set()

    async def latest(self, n_samples: int) -> tuple[np.ndarray, int]:
        """Return the newest window and cumulative sample counter.

        Receives:
        - `n_samples`: desired window length from tail of ring.

        Returns:
        - `(window, total_samples_seen)`.
        """

        async with self.lock:
            available = self._fill
            want = min(n_samples, available)
            if want == 0:
                return np.zeros((0,), dtype=np.float32), self.total_samples

            # The newest 'want' samples end at _write_pos (exclusive)
            start = (self._write_pos - want) % self.max_samples
            if start + want <= self.max_samples:
                out = self._ring[start:start + want].copy()
            else:
                # Wraps around the boundary
                first = self.max_samples - start
                out = np.concatenate([
                    self._ring[start:],
                    self._ring[:want - first],
                ])
            return out, self.total_samples

    async def wait_for_total_after(self, total_samples: int, timeout: float) -> bool:
        """Wait until the cumulative sample counter advances past `total_samples`.

        This clears the event only while holding the buffer lock and only after
        confirming no newer samples are present, which avoids missed wakeups
        between a caller's `latest()` read and a subsequent `Event.clear()`.

        Receives:
        - `total_samples`: last observed cumulative sample count.
        - `timeout`: max seconds to wait.

        Returns:
        - `True` if newer samples arrived before timeout, else `False`.
        """

        while True:
            async with self.lock:
                if self.total_samples > total_samples:
                    return True
                self.new_data_event.clear()
            try:
                await asyncio.wait_for(self.new_data_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return False


class StreamingLinearResampler:
    """Low-latency stateful linear resampler for chunked mono PCM streams.

    This avoids per-chunk filter resets that can introduce audible ticks or
    feature glitches when short websocket PCM packets are resampled
    independently.
    """

    def __init__(self, src_rate: int, dst_rate: int):
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.step = float(self.src_rate) / float(self.dst_rate)
        self._buf = np.zeros((0,), dtype=np.float32)
        self._next_pos = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return np.zeros((0,), dtype=np.float32)

        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._buf.size == 0:
            self._buf = chunk.copy()
        else:
            self._buf = np.concatenate([self._buf, chunk])

        # Need at least two points for interpolation.
        if self._buf.size < 2:
            return np.zeros((0,), dtype=np.float32)

        out = []
        max_pos = float(self._buf.size - 1)
        while self._next_pos < max_pos:
            left = int(self._next_pos)
            frac = self._next_pos - left
            right = left + 1
            sample = self._buf[left] * (1.0 - frac) + self._buf[right] * frac
            out.append(sample)
            self._next_pos += self.step

        keep_from = max(0, int(self._next_pos) - 1)
        if keep_from > 0:
            self._buf = self._buf[keep_from:]
            self._next_pos -= keep_from

        if not out:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)


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
        self._upsampler_24k_to_48k = StreamingLinearResampler(src_rate=24000, dst_rate=48000)

    async def append_from_24k(self, mono24k: np.ndarray) -> None:
        """Append 24kHz mono PCM by upsampling to 48kHz.

        Receives:
        - `mono24k`: float PCM at 24kHz.

        Returns:
        - `None`.
        """

        if mono24k.size == 0:
            return
        mono48k = self._upsampler_24k_to_48k.process(mono24k.astype(np.float32, copy=False))
        if mono48k.size == 0:
            return
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
    """FIFO BGR frame buffer with last-frame fallback for pacing."""

    def __init__(self, maxsize: int = 64):
        """Create a bounded FIFO buffer for generated avatar frames.

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

    async def get_nowait(self) -> np.ndarray:
        """Return the oldest frame in queue or last_frame if empty (non-blocking).

        Receives:
        - None.

        Returns:
        - BGR frame (`np.ndarray`).
        """

        try:
            frame = self.queue.get_nowait()
            self.last_frame = frame
            return frame
        except asyncio.QueueEmpty:
            # Diagnostic log (Claude Recommendation)
            if self.last_frame is not None:
                # print("[BUFFER] get_nowait: queue empty, returning last_frame")
                pass
            else:
                 print("[BUFFER] get_nowait: queue empty, NO LAST FRAME")
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
