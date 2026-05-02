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
        self._start_epoch = None
        self._frames_sent_counter = 0

    async def recv(self):
        """Produce the next `av.VideoFrame` for WebRTC sender with rigid wall-clock pacing.

        Receives:
        - None.

        Returns:
        - `av.VideoFrame` with timestamps from strict wall-clock.
        """

        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc/av is not installed")
        
        import time as _time
        _recv_called_at = _time.perf_counter()
        queue_depth = self.buffer.queue.qsize()

        now = time.time()

        # --- Jitter Buffer Cushion (Claude Recommendation) ---
        # Wait for an initial pool of frames before starting the real-time clock.
        # This adds ~200ms latency but ensures the pacer has a sequence to drain.
        if self._start_epoch is None:
            if queue_depth < 5:
                # Return the idle frame until the cushion is ready
                frame_bgr = await self.buffer.get_nowait()
                video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                video_frame.pts = 0
                video_frame.time_base = fractions.Fraction(1, 90000)
                return video_frame
            
            self._start_epoch = now
            self._frames_sent_counter = 0
            print(f"[PACE] Jitter buffer ready (depth={queue_depth}), starting pacer clock")

        # Calculate target delivery time for this frame index
        target_time = self._start_epoch + (self._frames_sent_counter * self.frame_interval)
        
        # Rigid pacing: wait for the next 40ms boundary
        wait = target_time - now
        if wait > 0:
            await asyncio.sleep(wait)
        elif wait < -self.frame_interval: 
            # Systemic lag detected (e.g. event loop starvation)
            # Slide the epoch forward to match current reality, but keep the counter/PTS climbing.
            # This prevents a burst of zero-wait packets while maintaining monotonic timestamps.
            missed = int(-wait / self.frame_interval)
            self._start_epoch += missed * self.frame_interval
            print(f"[PACE] *** EPOCH RESET *** lag={-wait*1000:.1f}ms counter={self._frames_sent_counter} queue={queue_depth}")

        slept_ms = (_time.perf_counter() - _recv_called_at) * 1000
        if slept_ms > 50: # budget is 40ms
            print(f"[PACE] SLOW recv: slept={slept_ms:.1f}ms queue={queue_depth} wait={wait*1000:.1f}ms counter={self._frames_sent_counter}")

        # Fetch from FIFO queue (non-blocking)
        frame_bgr = await self.buffer.get_nowait()
        
        if frame_bgr is self.buffer.last_frame and queue_depth == 0:
             if self._frames_sent_counter > 0:
                 print(f"[PACE] REPEAT last_frame (queue was empty) counter={self._frames_sent_counter}")

        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        
        # pts must increase based on the 90kHz clock for smooth decoding
        video_frame.pts = int(self._frames_sent_counter * (90000 / self.fps))
        video_frame.time_base = fractions.Fraction(1, 90000)
        
        self._frames_sent_counter += 1
        self.stats["video_frames_sent"] = self._frames_sent_counter
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

