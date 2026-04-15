"""Dataclasses for runtime args and per-session state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class AppArgs:
    """Typed runtime configuration produced by CLI parsing.

    Receives:
    - Parsed command-line values in `parse_args()`.

    Returns:
    - Dataclass instance consumed by `WebRtcApp` and engine/client components.

    Important:
    - Carries signaling, session lifecycle, Personaplex, and MuseTalk knobs.
    """

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
    avatar_fps: int
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
    musetalk_only: bool
    enable_api_auth: bool
    api_token: str
    session_offer_timeout_seconds: float
    session_max_age_seconds: float
    session_cleanup_interval_seconds: float
    single_session_mode: bool
    web_test_only: bool
    debug: bool
    debug_events_limit: int


@dataclass
class SessionState:
    """In-memory state for one browser signaling/media session.

    Receives:
    - Created in `WebRtcApp._create_session()`, then mutated through lifecycle.

    Returns:
    - Dataclass that stores peer, token, stats, and Personaplex bridge handles.

    Important:
    - In single-session mode, this object is replaced when a new offer arrives.
    """

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
