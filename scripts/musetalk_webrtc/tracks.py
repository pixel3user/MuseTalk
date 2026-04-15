"""aiortc media track adapters for MuseTalk video/audio buffers."""

import asyncio
import fractions
import time

import numpy as np

from .buffers import AudioTrackBuffer, VideoFrameBuffer
from .rtc import AIORTC_AVAILABLE, AudioStreamTrack, MediaStreamError, VideoStreamTrack, av

class MuseTalkVideoTrack(VideoStreamTrack):
    """aiortc video track that streams frames from `VideoFrameBuffer`."""

    def __init__(self, buffer: VideoFrameBuffer, fps: int, stats: dict):
        """Initialize video track wrapper.

        Receives:
        - `buffer`: shared frame source.
        - `fps`: pacing target.
        - `stats`: mutable telemetry dict.

        Returns:
        - `None`.
        """

        super().__init__()
        self.buffer = buffer
        self.fps = max(1, fps)
        self.frame_interval = 1.0 / self.fps
        self.stats = stats

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Override aiortc 30fps default to pace exactly to the requested engine FPS."""
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += int((1.0 / self.fps) * 90000)
            wait = self._start + (self._timestamp / 90000) - time.time()
            await asyncio.sleep(max(0.0, wait))
        else:
            self._start = time.time()
            self._timestamp = 0

        return self._timestamp, fractions.Fraction(1, 90000)

    async def recv(self):
        """Produce the next `av.VideoFrame` for WebRTC sender.

        Receives:
        - None.

        Returns:
        - `av.VideoFrame` with timestamps from aiortc clock.
        """

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
    """aiortc audio track that streams mono 48k frames from `AudioTrackBuffer`."""

    def __init__(self, audio_buffer: AudioTrackBuffer, stats: dict):
        """Initialize audio track wrapper.

        Receives:
        - `audio_buffer`: shared 48k sample buffer.
        - `stats`: mutable telemetry dict.

        Returns:
        - `None`.
        """

        super().__init__()
        self.audio_buffer = audio_buffer
        self.samples_per_frame = 960  # 20ms @ 48kHz
        self.sample_rate = 48000
        self.stats = stats

    async def recv(self):
        """Produce next 20ms mono audio frame for WebRTC sender.

        Receives:
        - None.

        Returns:
        - `av.AudioFrame` (`s16`, `mono`, 48kHz).
        """

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

