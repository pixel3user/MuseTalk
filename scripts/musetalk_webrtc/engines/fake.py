"""GPU-free fake inference engine for testing the full pipeline.

This engine satisfies the InferenceEngine protocol without importing torch,
enabling testing of:
- Audio windowing and hop logic
- Video frame publishing and pacing
- Ring buffer interactions
- Session lifecycle and backpressure behavior
- Failure injection and recovery

Usage:
    from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

    engine = FakeInferenceEngine(args, pcm_ring_16k, video_buffer)
    task = asyncio.create_task(engine.run())
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import cv2
import numpy as np

from ..buffers import PcmRingBuffer, VideoFrameBuffer
from ..models import AppArgs


class FakeInferenceEngine:
    """Drop-in replacement for MuseTalkRealtimeEngine. No torch, no CUDA.

    Behavior contract (matches real engine):
    - Reads from pcm_ring_16k using the same windowing logic
    - Publishes to video_buffer at args.fps when audio energy > threshold
    - Top-up duplicates during silence
    - Configurable simulated inference latency

    Diverges from real engine:
    - Frames are synthetic (colorbar + RMS-driven mouth rectangle)
    - No VAD model (uses simple energy gate with configurable threshold)
    - No mouth smoothing (deterministic outputs for assertions)
    """

    def __init__(
        self,
        args: AppArgs,
        pcm_ring_16k: PcmRingBuffer,
        video_buffer: VideoFrameBuffer,
        *,
        simulated_inference_ms: float = 30.0,
        fail_after_n_jobs: int = -1,
        energy_threshold: float = 0.01,
        frame_height: int = 720,
        frame_width: int = 1280,
    ):
        """Create a fake inference engine.

        Receives:
        - `args`: same AppArgs as real engine (uses window_ms, hop_ms, fps, etc).
        - `pcm_ring_16k`: audio input ring buffer.
        - `video_buffer`: output frame queue for WebRTC video track.
        - `simulated_inference_ms`: artificial delay per inference job (models GPU latency).
        - `fail_after_n_jobs`: if >= 0, raise RuntimeError after N successful jobs.
        - `energy_threshold`: RMS threshold below which audio is treated as silence.

        Returns:
        - `None`.
        """
        self.args = args
        self.pcm_ring = pcm_ring_16k
        self.video_buffer = video_buffer
        self.simulated_inference_ms = simulated_inference_ms
        self.fail_after_n_jobs = fail_after_n_jobs
        self.energy_threshold = energy_threshold

        self.stop_event = asyncio.Event()
        self.ready = asyncio.Event()
        self.last_total_samples = -1
        self.last_publish_epoch = 0.0
        self.last_error = ""
        self.jobs = 0
        self.dropped_audio_ms_total = 0.0
        self.frames_published = 0
        self.topup_frames = 0

        self._h = frame_height
        self._w = frame_width
        self._idle_frame = self._render_idle_frame()
        self.video_buffer.last_frame = self._idle_frame.copy()

    def _render_idle_frame(self) -> np.ndarray:
        """Render the initial idle frame (shown before first audio arrives)."""
        frame = np.full((self._h, self._w, 3), 50, dtype=np.uint8)
        cv2.putText(
            frame,
            "FakeEngine: idle",
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (200, 200, 200),
            2,
        )
        return frame

    def _render_frame(self, rms: float, frame_no: int) -> np.ndarray:
        """Render a synthetic avatar frame with RMS-driven mouth.

        The mouth rectangle height is proportional to audio energy,
        making it easy to verify in tests that speech produces animation.
        """
        frame = np.full((self._h, self._w, 3), 60, dtype=np.uint8)
        # Mouth: rectangle that opens proportionally to RMS
        mouth_h = int(np.clip(rms * 400, 4, 80))
        cx, cy = self._w // 2, self._h // 2 + 40
        cv2.rectangle(
            frame,
            (cx - 50, cy - mouth_h // 2),
            (cx + 50, cy + mouth_h // 2),
            (40, 40, 220),
            -1,
        )
        # Info overlay
        cv2.putText(
            frame,
            f"FAKE rms={rms:.4f} job={self.jobs} f={frame_no}",
            (40, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return frame

    async def _topup_video_queue(self, target_frames: int) -> None:
        """Keep a small queue of duplicate frames during audio stalls."""
        target = max(1, min(int(target_frames), int(self.args.video_queue_size)))
        current = self.video_buffer.queue.qsize()
        if current >= target:
            return
        frame = self.video_buffer.last_frame
        for _ in range(target - current):
            await self.video_buffer.publish(frame)
            self.last_publish_epoch = time.time()
            self.topup_frames += 1

    async def run(self) -> None:
        """Main inference loop — mirrors real engine's windowing logic.

        1. Wait for enough new audio in pcm_ring (hop_ms worth of samples).
        2. Simulate inference delay.
        3. Publish N frames proportional to the audio duration consumed.
        4. Repeat until stop_event is set.
        """
        window_samples = int((self.args.window_ms / 1000.0) * 16000)
        min_advance_samples = int((self.args.hop_ms / 1000.0) * 16000)
        max_advance_samples = int((self.args.max_advance_ms / 1000.0) * 16000)

        self.ready.set()

        while not self.stop_event.is_set():
            window, total = await self.pcm_ring.latest(window_samples)

            if self.last_total_samples < 0:
                new_samples = int(window.size)
            else:
                new_samples = max(0, int(total - self.last_total_samples))

            # Not enough new audio — top up and wait
            if new_samples < min_advance_samples:
                await self._topup_video_queue(target_frames=max(6, self.args.batch_size))
                await self.pcm_ring.wait_for_total_after(total, timeout=0.1)
                continue

            self.last_total_samples = total

            # Cap advance (mirrors real engine's dropped audio logic)
            if max_advance_samples > 0 and new_samples > max_advance_samples:
                dropped = (new_samples - max_advance_samples) * 1000.0 / 16000.0
                self.dropped_audio_ms_total += dropped
                new_samples = max_advance_samples

            # Failure injection
            if self.fail_after_n_jobs >= 0 and self.jobs >= self.fail_after_n_jobs:
                self.last_error = "FakeEngine: simulated failure after job limit"
                raise RuntimeError(self.last_error)

            # Simulate inference time
            if self.simulated_inference_ms > 0:
                await asyncio.sleep(self.simulated_inference_ms / 1000.0)

            # Compute output
            new_audio = window[-new_samples:]
            rms = float(np.sqrt(np.mean(new_audio**2))) if new_audio.size > 0 else 0.0

            n_frames = max(1, int(round((new_samples / 16000.0) * self.args.fps)))
            n_frames = min(n_frames, max(1, self.args.max_tail_frames))

            # Only publish animated frames if audio energy exceeds threshold
            if rms >= self.energy_threshold:
                for i in range(n_frames):
                    frame = self._render_frame(rms, self.frames_published)
                    await self.video_buffer.publish(frame)
                    self.last_publish_epoch = time.time()
                    self.frames_published += 1
            else:
                # Silence: publish one idle frame to keep queue healthy
                await self.video_buffer.publish(self._idle_frame)
                self.last_publish_epoch = time.time()
                self.frames_published += 1

            self.jobs += 1

    def status(self) -> dict:
        """Return engine diagnostics (same shape as real engine)."""
        return {
            "jobs": self.jobs,
            "last_publish_epoch": self.last_publish_epoch or None,
            "last_error": self.last_error or None,
            "dropped_audio_ms_total": round(self.dropped_audio_ms_total, 1),
            "frames_published": self.frames_published,
            "topup_frames": self.topup_frames,
            "fake": True,
            "ready": self.ready.is_set(),
            "simulated_inference_ms": self.simulated_inference_ms,
        }
