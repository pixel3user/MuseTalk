# `scripts/musetalk_webrtc/*` Reference

The implementation is now modular:
- entrypoint shim: `scripts/personaplex_musetalk_webrtc.py`
- core package: `scripts/musetalk_webrtc/`

This document describes the same runtime behavior, now split across modules.
It focuses on:
- what each symbol receives,
- what it returns,
- and important runtime behavior.

## Module Map

- `scripts/musetalk_webrtc/web_ui.py`: embedded browser HTML/JS.
- `scripts/musetalk_webrtc/constants.py`: shared constants (token header).
- `scripts/musetalk_webrtc/models.py`: `AppArgs`, `SessionState`.
- `scripts/musetalk_webrtc/buffers.py`: PCM/video buffer primitives.
- `scripts/musetalk_webrtc/personaplex_io.py`: mirror + duplex chat websocket clients.
- `scripts/musetalk_webrtc/engine.py`: MuseTalk realtime inference engine.
- `scripts/musetalk_webrtc/tracks.py`: aiortc audio/video track wrappers.
- `scripts/musetalk_webrtc/server.py`: `WebRtcApp` routes/session/signaling logic.
- `scripts/musetalk_webrtc/cli.py`: CLI parsing and app boot.

## How To Use This Doc + Inline Docs

The Python file now includes inline docstrings on every class and function.

Use this guide like this:
1. Start with this markdown for the high-level map (routes + component roles).
2. Open the source and read each function docstring for exact inputs/outputs.
3. Use `/offer` flow for quickest debugging path:
   - browser `start()` in `HTML_PAGE`,
   - `POST /offer` route registration in `build_app()`,
   - `offer()` handler,
   - `_handle_offer()` negotiation + track wiring,
   - `_consume_inbound_audio()` routing to Personaplex / local rings.

Docstring style used in code:
- `Receives`: key arguments and expected shape/type.
- `Returns`: response type/value.
- `Important`: side effects, lifecycle behavior, and mode-specific logic.

## Route Map

Routes are registered in `WebRtcApp.build_app()`.

| Route | Method | Handler | Purpose |
|---|---|---|---|
| `/` | GET | `index` | Web preview page (embedded JS client). |
| `/healthz` | GET | `healthz` | Basic liveness + session summary. |
| `/config` | GET | `config` | Browser RTCPeerConnection ICE config. |
| `/status` | GET | `status` | Full diagnostics (mirror/engine/webrtc/personaplex). |
| `/snapshot.jpg` | GET | `snapshot` | Latest frame snapshot (JPEG). |
| `/offer` | POST | `offer` | Legacy SDP offer endpoint; auto-creates a session. |
| `/v1/config` | GET | `config_v1` | API metadata + token header name. |
| `/v1/session` | GET | `active_session` | Active session summary. |
| `/v1/sessions` | POST | `create_session` | Creates a session and token. |
| `/v1/sessions/{session_id}/offer` | POST | `session_offer` | Session-bound SDP offer endpoint. |
| `/v1/sessions/{session_id}/stats` | GET | `session_stats` | Session stats (requires session token header). |
| `/v1/sessions/{session_id}` | DELETE | `delete_session` | Session teardown. |

---

## Embedded Browser JS Functions (`HTML_PAGE`)

### `waitIceGatheringComplete(pc, timeoutMs=4000)`
- Receives: browser `RTCPeerConnection`, timeout.
- Returns: `Promise<void>`.
- Important: waits non-trickle ICE gathering before posting offer.

### `cleanupPeer()`
- Receives: none (uses global `pc`, `localStream`).
- Returns: `Promise<void>`.
- Important: stops local media tracks and closes browser peer connection.

### `cleanupSession()`
- Receives: none (uses global `sessionId`, `sessionToken`).
- Returns: `Promise<void>`.
- Important: calls `DELETE /v1/sessions/{id}` with `x-session-token` when available.

### `start()`
- Receives: none (triggered by Start button).
- Returns: `Promise<void>`.
- Important:
  - requests microphone,
  - builds browser offer,
  - `POST /offer`,
  - applies returned answer SDP,
  - stores `session_id` + optional session token.

### `stop()`
- Receives: none (triggered by Stop button).
- Returns: `Promise<void>`.
- Important: closes peer/session and resets UI state.

---

## Data Classes

### `AppArgs`
- Receives: values from CLI parser (`parse_args`).
- Returns: dataclass instance containing runtime config.
- Important fields:
  - Network/signaling: `host`, `port`, ICE fields.
  - Personaplex integration: `personaplex_*`.
  - MuseTalk runtime: model paths, fps, batch size, smoothing, etc.
  - Session behavior: `session_offer_timeout_seconds`, `session_max_age_seconds`, `single_session_mode`.
  - Feature toggles: `web_test_only`, `enable_api_auth`, `webrtc_audio_loopback`.

### `SessionState`
- Receives: created in `_create_session` and updated during call lifecycle.
- Returns: dataclass instance tracking one signaling/media session.
- Important fields:
  - WebRTC state: `pc`, `pcid`, `inbound_task`, `last_offer_epoch`.
  - Mic stats: `mic_frames_rx`, `mic_samples_rx_16k`, `last_mic_rx_epoch`.
  - Personaplex bridge stats: `personaplex_connected`, `personaplex_audio_frames_tx/rx`, `personaplex_last_rx_epoch`.
  - Auth/lifecycle: `token`, `created_epoch`, `close_reason`.

---

## Buffer Classes

### `class PcmRingBuffer`

#### `__init__(max_samples: int)`
- Receives: max samples to retain.
- Returns: `None`.
- Important: stores float32 mono PCM with asyncio lock.

#### `append(samples: np.ndarray) -> None`
- Receives: mono PCM samples.
- Returns: `None`.
- Important: appends and clips to ring size.

#### `latest(n_samples: int) -> tuple[np.ndarray, int]`
- Receives: desired window length.
- Returns: `(window_samples, total_samples_seen)`.
- Important: used by inference loop to compute incremental advancement.

### `class AudioTrackBuffer`

#### `__init__(max_samples_48k: int)`
- Receives: max buffered 48k samples.
- Returns: `None`.

#### `append_from_24k(mono24k: np.ndarray) -> None`
- Receives: 24k mono float PCM.
- Returns: `None`.
- Important: upsamples 24k->48k by duplication for outbound WebRTC audio track.

#### `pop_48k(n_samples: int) -> np.ndarray`
- Receives: requested 48k frame size.
- Returns: exactly `n_samples` (zero-padded if underflow).
- Important: drives `MuseTalkAudioTrack.recv()`.

### `class VideoFrameBuffer`

#### `__init__(maxsize: int)`
- Receives: queue size.
- Returns: `None`.

#### `publish(frame_bgr: np.ndarray) -> None`
- Receives: BGR frame.
- Returns: `None`.
- Important: drops oldest frame when full; keeps `last_frame` fallback.

#### `get(timeout: float=0.12) -> np.ndarray`
- Receives: timeout.
- Returns: next frame or `last_frame` on timeout.
- Important: ensures video continues even when no new inference frame arrives.

#### `snapshot_jpeg() -> bytes`
- Receives: none.
- Returns: JPEG bytes (or empty bytes on encoding failure).

---

## Personaplex IO Clients

### `class PersonaPlexMirrorClient`

#### `__init__(ws_url, pcm_ring_24k, pcm_ring_16k, audio_track_buffer, status_json, reconnect_delay_seconds)`
- Receives: mirror websocket URL and output buffers.
- Returns: `None`.
- Important: designed for `/api/avatar/audio` read-only mirror mode.

#### `_write_status() -> None`
- Receives: none.
- Returns: `None`.
- Important: writes mirror diagnostics JSON when enabled.

#### `run() -> None`
- Receives: none.
- Returns: coroutine result `None`.
- Important:
  - connects/reconnects websocket,
  - decodes Opus packets,
  - fills 24k ring, 16k ring, and outbound audio buffer.

### `class PersonaPlexChatBridge`

#### `__init__(ws_url, session, pcm_ring_24k, pcm_ring_16k, audio_track_buffer, reconnect_delay_seconds)`
- Receives: `/api/chat` websocket URL, owning `SessionState`, and output buffers.
- Returns: `None`.
- Important: this is the per-session duplex bridge created when a browser session starts.

#### `push_uplink_pcm24k(pcm24k: np.ndarray) -> None`
- Receives: browser mic audio resampled to 24k.
- Returns: `None`.
- Important: queues uplink audio to send into Personaplex websocket.

#### `_recv_loop(ws) -> None`
- Receives: open websocket.
- Returns: coroutine result `None`.
- Important:
  - handles handshake kind `0`, audio kind `1`,
  - decodes opus,
  - feeds MuseTalk rings + outbound audio track,
  - updates session RX stats.

#### `_send_loop(ws) -> None`
- Receives: open websocket.
- Returns: coroutine result `None`.
- Important: encodes queued 24k mic PCM into Opus and sends kind `1` frames.

#### `run() -> None`
- Receives: none.
- Returns: coroutine result `None`.
- Important: manages websocket lifecycle/reconnect and concurrent send/recv loops.

---

## MuseTalk Inference Engine

### `class MuseTalkRealtimeEngine`

#### `__init__(args, pcm_ring_16k, video_buffer)`
- Receives: runtime args, 16k input ring, output video buffer.
- Returns: `None`.
- Important: imports and initializes heavy MuseTalk runtime/models.

#### `_setup_runtime() -> None`
- Receives: none.
- Returns: `None`.
- Important:
  - configures model runtime namespace,
  - loads VAE/UNet/PE/Whisper,
  - applies fp16/fp32 selection,
  - initializes face parsing mode.

#### `_infer_window_frames(pcm16k_window, new_frames) -> list[np.ndarray]`
- Receives: latest audio window and number of new frames to publish.
- Returns: list of blended avatar BGR frames.
- Important: does audio feature extraction + UNet inference + blending + temporal smoothing.

#### `run() -> None`
- Receives: none.
- Returns: coroutine result `None`.
- Important: main inference loop with overlap windowing and tail-frame publishing.

#### `status() -> dict`
- Receives: none.
- Returns: engine counters/state.

---

## WebRTC Media Track Wrappers

### `class MuseTalkVideoTrack(VideoStreamTrack)`

#### `__init__(buffer, fps, stats)`
- Receives: `VideoFrameBuffer`, target fps, shared stats dict.
- Returns: `None`.

#### `recv()`
- Receives: none.
- Returns: `av.VideoFrame`.
- Important: pulls frame from buffer and timestamps it via aiortc `next_timestamp()`.

### `class MuseTalkAudioTrack(AudioStreamTrack)`

#### `__init__(audio_buffer, stats)`
- Receives: `AudioTrackBuffer`, shared stats dict.
- Returns: `None`.

#### `recv()`
- Receives: none.
- Returns: `av.AudioFrame` (mono 48k, 20ms).
- Important: maintains its own audio clock and zero-pads on buffer underrun.

---

## Web App + Signaling Core

### `class WebRtcApp`

#### `__init__(args)`
- Receives: `AppArgs`.
- Returns: `None`.
- Important:
  - allocates shared buffers and stats,
  - builds either mirror client or chat-bridge-capable runtime,
  - constructs heavy engine unless `web_test_only`.

#### `_personaplex_chat_enabled() -> bool`
- Receives: none.
- Returns: true when non-test mode and `personaplex_path` ends with `/api/chat`.

#### `_seed_web_test_frame() -> None`
- Receives: none.
- Returns: `None`.
- Important: provides placeholder frame when `--web-test-only` skips model load.

#### `_coerce_bool(value, default=False) -> bool`
- Receives: arbitrary value.
- Returns: parsed boolean.

#### `_safe_json(request) -> dict`
- Receives: aiohttp request.
- Returns: parsed JSON dict or `{}`.

#### `_touch_session(session) -> None`
- Receives: `SessionState`.
- Returns: `None`.
- Important: updates `last_activity_epoch`.

#### `_session_state(session) -> str`
- Receives: `SessionState`.
- Returns: pc state (`new` fallback).

#### `_session_should_expire(session, now) -> tuple[bool, str]`
- Receives: session + current epoch.
- Returns: `(expire?, reason)`.
- Important: enforces max age, offer timeout, and stale closed/failure cleanup.

#### `_close_session(session, reason) -> None`
- Receives: session and reason string.
- Returns: coroutine result `None`.
- Important:
  - stops Personaplex bridge/task,
  - cancels inbound audio task,
  - closes RTCPeerConnection,
  - removes from in-memory session map.

#### `_expire_stale_sessions() -> None`
- Receives: none.
- Returns: coroutine result `None`.
- Important: scans all sessions and closes expired ones.

#### `_session_cleanup_loop() -> None`
- Receives: none.
- Returns: coroutine result `None`.
- Important: periodic background stale-session cleanup.

#### `_aiortc_ice_servers()`
- Receives: none.
- Returns: list of `RTCIceServer` objects.
- Important: applies TURN credentials only to `turn:`/`turns:` urls.

#### `_browser_rtc_config() -> dict`
- Receives: none.
- Returns: browser-friendly RTC config dict.

#### `_build_mirror_ws_url() -> str`
- Receives: none.
- Returns: websocket URL built from `personaplex_*` args.
- Important: includes query params (`text_prompt`, `voice_prompt`, extra query pairs).

#### `on_startup(_app) -> None`
- Receives: aiohttp app.
- Returns: coroutine result `None`.
- Important: starts mirror client (if enabled), engine (if enabled), and cleanup task.

#### `on_cleanup(_app) -> None`
- Receives: aiohttp app.
- Returns: coroutine result `None`.
- Important: cancels tasks, closes all sessions/peers.

#### `index(_request) -> web.Response`
- Receives: request.
- Returns: HTML page response.

#### `config(_request) -> web.Response`
- Receives: request.
- Returns: JSON RTC config for browser client.

#### `status(_request) -> web.Response`
- Receives: request.
- Returns: diagnostics JSON.
- Important: includes mode, personaplex mode flags, mirror/engine metrics, peer/session counts.

#### `snapshot(_request) -> web.Response`
- Receives: request.
- Returns: JPEG response or `503` if unavailable.

#### `healthz(_request) -> web.Response`
- Receives: request.
- Returns: lightweight health JSON.

#### `config_v1(_request) -> web.Response`
- Receives: request.
- Returns: API metadata (`session_token_header`, auth flags, RTC config).

#### `active_session(_request) -> web.Response`
- Receives: request.
- Returns: active session payload or `null`.

#### `_active_session() -> Optional[SessionState]`
- Receives: none.
- Returns: active session object if found.

#### `_create_session() -> SessionState`
- Receives: none.
- Returns: newly created session with token/id/timestamps.

#### `_get_session(session_id) -> Optional[SessionState]`
- Receives: session id string.
- Returns: session object or `None`.

#### `_session_conflict_payload(session) -> dict`
- Receives: active session.
- Returns: 409-friendly payload explaining replacement behavior.

#### `create_session(request) -> web.Response`
- Receives: request body/query (supports `replace=true`).
- Returns: JSON with `session_id`, `token`, metadata.
- Important: enforces single-session mode unless replaced.

#### `_check_session_token(request, session) -> Optional[web.Response]`
- Receives: request and session.
- Returns: `None` when valid; JSON error response when missing/invalid.

#### `delete_session(request) -> web.Response`
- Receives: session-id route param + token header.
- Returns: teardown confirmation JSON.

#### `session_stats(request) -> web.Response`
- Receives: session-id + token header.
- Returns: session stats payload.

#### `_session_payload(session) -> dict`
- Receives: session object.
- Returns: normalized stats dict (mic + personaplex + lifecycle fields).

#### `_start_personaplex_chat_bridge(session) -> None`
- Receives: session object.
- Returns: coroutine result `None`.
- Important:
  - active only for `/api/chat` mode,
  - creates per-session `PersonaPlexChatBridge` task,
  - binds bridge metrics to session state.

#### `offer(request) -> web.Response`
- Receives: legacy SDP offer body.
- Returns: SDP answer JSON + auto-created session id (+ token for legacy flow).
- Important: this is what browser JS `fetch('/offer')` calls.

#### `session_offer(request) -> web.Response`
- Receives: session-id route + token header + SDP offer body.
- Returns: SDP answer JSON for that session.

#### `_handle_offer(request, session, include_session_token=False) -> web.Response`
- Receives: request JSON (`type=offer`, `sdp`) and session object.
- Returns: answer JSON.
- Important:
  - creates/attaches RTCPeerConnection,
  - starts Personaplex chat bridge in chat mode,
  - installs two local callbacks:
    - `on_connectionstatechange`: peer state update/cleanup,
    - `on_track`: starts inbound audio consume task.

#### `_consume_inbound_audio(track, session) -> None`
- Receives: inbound aiortc audio track + session.
- Returns: coroutine result `None` (loop until closed).
- Important:
  - converts/resamples browser mic audio,
  - updates mic stats,
  - if chat bridge active: forwards mic to Personaplex uplink queue,
  - otherwise uses existing local paths (`input_source webrtc/mixed`, optional loopback).

#### `_resample_audio(pcm, src_sr, dst_sr) -> np.ndarray`
- Receives: mono PCM and sample rates.
- Returns: float32 resampled PCM.

#### `auth_middleware(request, handler)`
- Receives: aiohttp request + downstream handler.
- Returns: downstream response or 401 JSON.
- Important: only guards `/v1/*` paths when `--enable-api-auth` is enabled.

#### `_wait_for_ice_gathering(pc, timeout) -> None`
- Receives: RTCPeerConnection + timeout seconds.
- Returns: coroutine result `None`.
- Important: waits for ICE gathering completion (or timeout).
  - local callback `_on_ice_state_change`: sets event when state becomes `complete`.

#### `build_app() -> web.Application`
- Receives: none.
- Returns: fully wired aiohttp app.

---

## Top-Level Functions

### `parse_args() -> AppArgs`
- Receives: CLI arguments.
- Returns: populated `AppArgs`.
- Important: centralizes all runtime toggles, including web test mode and personaplex settings.

### `main() -> None`
- Receives: none.
- Returns: `None`.
- Important:
  - validates aiortc availability,
  - warns when `/api/chat` path is used without voice prompt,
  - starts aiohttp app via `web.run_app`.
