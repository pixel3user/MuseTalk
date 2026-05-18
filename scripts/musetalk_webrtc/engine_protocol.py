"""Protocol definition for MuseTalk inference engines.

This module defines the contract that any inference engine must satisfy,
enabling GPU-free testing via FakeInferenceEngine while maintaining
production parity with the real MuseTalkRealtimeEngine.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable


@runtime_checkable
class InferenceEngine(Protocol):
    """Contract for inference engines consumed by WebRtcApp.

    Both the real MuseTalkRealtimeEngine (requires GPU) and FakeInferenceEngine
    (CPU-only, for testing) must satisfy this interface.

    Lifecycle:
    1. Constructed by an engine_factory callable.
    2. `run()` is started as an asyncio.Task by WebRtcApp.on_startup.
    3. Runs until `stop_event` is set during app cleanup.
    4. `status()` is polled by /status endpoint.
    """

    stop_event: asyncio.Event
    last_publish_epoch: float
    last_error: str
    jobs: int
    dropped_audio_ms_total: float

    async def run(self) -> None:
        """Main inference loop consuming audio from pcm_ring and publishing to video_buffer.

        Must respect self.stop_event: exit cleanly when set.
        Must not import torch at module level (lazy import only inside real engine).
        """
        ...

    def status(self) -> dict:
        """Return engine diagnostics dict for the /status endpoint.

        Required keys: jobs, last_publish_epoch, last_error, dropped_audio_ms_total.
        May include additional engine-specific keys.
        """
        ...
