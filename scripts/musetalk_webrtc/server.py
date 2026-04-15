"""Aiohttp signaling app: routes, sessions, and media wiring."""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlencode

import cv2
import numpy as np
from aiohttp import web
from scipy.signal import resample_poly

from .buffers import AudioTrackBuffer, PcmRingBuffer, VideoFrameBuffer
from .constants import SESSION_TOKEN_HEADER
from .engine import MuseTalkRealtimeEngine
from .models import AppArgs, SessionState
from .personaplex_io import PersonaPlexChatBridge, PersonaPlexMirrorClient
from .rtc import (
    AIORTC_AVAILABLE,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from .tracks import MuseTalkAudioTrack, MuseTalkVideoTrack
from .web_ui import HTML_PAGE

class WebRtcApp:
    """Aiohttp app state container for signaling, sessions, and media bridges."""

    def __init__(self, args: AppArgs):
        """Build buffers, optional clients, and runtime state.

        Receives:
        - `args`: validated runtime config.

        Returns:
        - `None`.
        """

        self.args = args
        self.started_epoch = time.time()
        self.pcs = set()
        self.pc_states = {}
        self.sessions: dict[str, SessionState] = {}
        self.active_session_id: Optional[str] = None
        self.track_stats = {
            "video_frames_sent": 0,
            "audio_frames_sent": 0,
            "last_video_send_epoch": 0.0,
            "last_audio_send_epoch": 0.0,
        }
        self.debug_events: list[dict[str, Any]] = []
        self.debug_events_limit = max(25, int(args.debug_events_limit))

        ring_samples = int(args.ring_buffer_seconds * 24000)
        self.pcm_ring_24k = PcmRingBuffer(max_samples=ring_samples)
        self.pcm_ring_16k = PcmRingBuffer(max_samples=int(args.ring_buffer_seconds * 16000))
        self.audio_track_buffer = AudioTrackBuffer(max_samples_48k=int(args.ring_buffer_seconds * 48000))
        self.video_buffer = VideoFrameBuffer(maxsize=args.video_queue_size)

        self.mirror_client: Optional[PersonaPlexMirrorClient] = None
        self.engine: Optional[MuseTalkRealtimeEngine] = None
        if not args.web_test_only:
            if (not args.musetalk_only) and (not self._personaplex_chat_enabled()):
                ws_url = self._build_mirror_ws_url()
                self.mirror_client = PersonaPlexMirrorClient(
                    ws_url=ws_url,
                    pcm_ring_24k=self.pcm_ring_24k,
                    pcm_ring_16k=self.pcm_ring_16k,
                    audio_track_buffer=self.audio_track_buffer,
                    status_json=args.status_json,
                    reconnect_delay_seconds=args.reconnect_delay_seconds,
                )
            self.engine = MuseTalkRealtimeEngine(
                args=args,
                pcm_ring_16k=self.pcm_ring_16k,
                video_buffer=self.video_buffer,
            )
        else:
            self._seed_web_test_frame()

        self.mirror_task: Optional[asyncio.Task] = None
        self.engine_task: Optional[asyncio.Task] = None
        self.session_cleanup_task: Optional[asyncio.Task] = None

    def _personaplex_chat_enabled(self) -> bool:
        """Return True when configured for `/api/chat` duplex mode."""

        if self.args.web_test_only or self.args.musetalk_only:
            return False
        return self.args.personaplex_path.rstrip("/").endswith("/api/chat")

    def _debug(self, event: str, **fields: Any) -> None:
        """Record and optionally print runtime debug events."""

        payload: dict[str, Any] = {
            "ts_epoch": round(time.time(), 3),
            "event": event,
        }
        if fields:
            payload.update(fields)
        self.debug_events.append(payload)
        if len(self.debug_events) > self.debug_events_limit:
            self.debug_events = self.debug_events[-self.debug_events_limit :]
        if self.args.debug:
            with contextlib.suppress(Exception):
                print(f"[debug] {json.dumps(payload, ensure_ascii=True, default=str)}")

    def _seed_web_test_frame(self) -> None:
        """Seed placeholder frame used when `--web-test-only` is enabled."""

        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = 34
        frame[:, :, 1] = 22
        frame[:, :, 2] = 18
        cv2.rectangle(frame, (40, 40), (w - 40, h - 40), (90, 160, 230), 2)
        cv2.putText(
            frame,
            "MuseTalk WebRTC Test Mode",
            (90, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (240, 240, 240),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "No model loading (connection test only)",
            (90, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        self.video_buffer.last_frame = frame

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        """Best-effort conversion for bool-like query/body values."""

        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    async def _safe_json(self, request: web.Request) -> dict:
        """Parse request JSON safely; return `{}` on any parse/shape error."""

        if not request.can_read_body:
            return {}
        with contextlib.suppress(Exception):
            body = await request.json()
            if isinstance(body, dict):
                return body
        return {}

    def _touch_session(self, session: SessionState) -> None:
        """Refresh session activity timestamp."""

        session.last_activity_epoch = time.time()

    def _session_state(self, session: SessionState) -> str:
        """Return current peer connection state for a session."""

        if session.pcid:
            return self.pc_states.get(session.pcid, "new")
        return "new"

    def _session_should_expire(self, session: SessionState, now: float) -> tuple[bool, str]:
        """Evaluate TTL/idle rules and return `(should_expire, reason)`."""

        age = now - session.created_epoch
        if age > max(1.0, self.args.session_max_age_seconds):
            return True, "max_age_exceeded"

        state = self._session_state(session)
        if session.pc is None and age > max(1.0, self.args.session_offer_timeout_seconds):
            return True, "offer_timeout"
        if state in {"closed", "failed", "disconnected"}:
            idle = now - session.last_activity_epoch
            if idle > max(5.0, self.args.session_cleanup_interval_seconds * 2.0):
                return True, f"pc_{state}"
        return False, ""

    async def _close_session(self, session: SessionState, reason: str) -> None:
        """Tear down bridge, tasks, peer, and remove session from registry."""

        sid = session.session_id
        session.close_reason = reason
        self._debug("session.close", session_id=sid, reason=reason, pc_state=self._session_state(session))
        if session.personaplex_bridge is not None:
            session.personaplex_bridge.stop_event.set()
        if session.personaplex_bridge_task is not None:
            session.personaplex_bridge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await session.personaplex_bridge_task
            session.personaplex_bridge_task = None
        session.personaplex_bridge = None
        session.personaplex_connected = False
        if session.inbound_task is not None:
            session.inbound_task.cancel()
            session.inbound_task = None
        if session.pc is not None:
            with contextlib.suppress(Exception):
                await session.pc.close()
            self.pcs.discard(session.pc)
            session.pc = None
        if session.pcid:
            self.pc_states[session.pcid] = "closed"
        self.sessions.pop(sid, None)
        if self.active_session_id == sid:
            self.active_session_id = None

    async def _expire_stale_sessions(self) -> None:
        """Close sessions that match expiration rules."""

        now = time.time()
        expired: list[tuple[SessionState, str]] = []
        for session in list(self.sessions.values()):
            should_expire, reason = self._session_should_expire(session, now)
            if should_expire:
                expired.append((session, reason))
        for session, reason in expired:
            await self._close_session(session, reason)

    async def _session_cleanup_loop(self) -> None:
        """Background periodic cleanup task for stale sessions."""

        interval = max(1.0, float(self.args.session_cleanup_interval_seconds))
        while True:
            await asyncio.sleep(interval)
            with contextlib.suppress(Exception):
                await self._expire_stale_sessions()

    def _aiortc_ice_servers(self):
        """Build aiortc ICE server objects from CLI config."""

        servers = []
        for url in self.args.ice_servers:
            if not str(url).strip():
                continue
            with contextlib.suppress(Exception):
                kw = {"urls": url}
                if self.args.ice_username and str(url).lower().startswith(("turn:", "turns:")):
                    kw["username"] = self.args.ice_username
                if self.args.ice_credential and str(url).lower().startswith(("turn:", "turns:")):
                    kw["credential"] = self.args.ice_credential
                servers.append(RTCIceServer(**kw))
        return servers

    def _browser_rtc_config(self) -> dict:
        """Build browser-facing RTC config JSON used by `/config`."""

        servers = []
        for url in self.args.ice_servers:
            if not str(url).strip():
                continue
            entry = {"urls": url}
            if self.args.ice_username and str(url).lower().startswith(("turn:", "turns:")):
                entry["username"] = self.args.ice_username
            if self.args.ice_credential and str(url).lower().startswith(("turn:", "turns:")):
                entry["credential"] = self.args.ice_credential
            servers.append(entry)
        return {"iceServers": servers, "iceTransportPolicy": self.args.ice_transport_policy}

    def _build_mirror_ws_url(self) -> str:
        """Construct PersonaPlex websocket URL with prompt/query parameters."""

        path = self.args.personaplex_path
        if not path.startswith("/"):
            path = "/" + path
        query = {
            "text_prompt": self.args.personaplex_text_prompt,
            "voice_prompt": self.args.personaplex_voice_prompt,
        }
        for kv in self.args.personaplex_extra_query:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            query[k] = v
        qs = urlencode(query)
        return f"ws://{self.args.personaplex_host}:{self.args.personaplex_port}{path}?{qs}"

    async def on_startup(self, _app: web.Application):
        """Aiohttp startup hook: start mirror/engine/cleanup tasks."""

        self._debug(
            "app.startup",
            port=self.args.port,
            input_source=self.args.input_source,
            personaplex_path=self.args.personaplex_path,
            chat_mode=self._personaplex_chat_enabled(),
        )
        if self.mirror_client is not None and self.args.input_source in ("mirror", "mixed"):
            self.mirror_task = asyncio.create_task(self.mirror_client.run())
        if self.engine is not None:
            self.engine_task = asyncio.create_task(self.engine.run())
        self.session_cleanup_task = asyncio.create_task(self._session_cleanup_loop())

    async def on_cleanup(self, _app: web.Application):
        """Aiohttp cleanup hook: stop tasks, close sessions, close peers."""

        self._debug("app.cleanup")
        if self.mirror_client is not None:
            self.mirror_client.stop_event.set()
        if self.engine is not None:
            self.engine.stop_event.set()
        for task in (self.mirror_task, self.engine_task, self.session_cleanup_task):
            if task is not None:
                task.cancel()
        for task in (self.mirror_task, self.engine_task, self.session_cleanup_task):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        for session in list(self.sessions.values()):
            with contextlib.suppress(Exception):
                await self._close_session(session, reason="shutdown")
        await asyncio.gather(*(pc.close() for pc in list(self.pcs)), return_exceptions=True)
        self.pcs.clear()
        self.sessions.clear()
        self.active_session_id = None

    async def index(self, _request: web.Request) -> web.Response:
        """Serve embedded browser test page."""

        return web.Response(text=HTML_PAGE, content_type="text/html")

    async def config(self, _request: web.Request) -> web.Response:
        """Return browser RTCPeerConnection config (`iceServers`, policy)."""

        return web.json_response(self._browser_rtc_config())

    async def status(self, _request: web.Request) -> web.Response:
        """Return deep runtime diagnostics for debugging and tuning."""

        await self._expire_stale_sessions()
        if self.mirror_client is None:
            mirror_payload = {
                "enabled": False,
                "connected": False,
                "packets": 0,
                "decoded_seconds": 0.0,
                "last_rx_epoch": None,
            }
        else:
            mirror_payload = {
                "enabled": True,
                "connected": self.mirror_client.connected,
                "packets": self.mirror_client.packets,
                "decoded_seconds": self.mirror_client.decoded_seconds,
                "last_rx_epoch": self.mirror_client.last_rx_epoch or None,
            }
        if self.engine is None:
            engine_payload = {
                "mode": "web_test_only",
                "jobs": 0,
                "last_publish_epoch": None,
                "last_error": None,
            }
        else:
            engine_payload = self.engine.status()
        payload = {
            "uptime_seconds": round(time.time() - self.started_epoch, 1),
            "mode": "web_test_only" if self.args.web_test_only else "musetalk",
            "personaplex": {
                "path": self.args.personaplex_path,
                "duplex_chat_mode": self._personaplex_chat_enabled(),
                "text_prompt_set": bool(self.args.personaplex_text_prompt),
                "voice_prompt": self.args.personaplex_voice_prompt or None,
            },
            "mirror": mirror_payload,
            "engine": engine_payload,
            "webrtc": {
                "peer_count": len(self.pcs),
                "peer_states": self.pc_states,
                "session_count": len(self.sessions),
                "active_session_id": self.active_session_id,
                "single_session_mode": self.args.single_session_mode,
                "track_stats": self.track_stats,
                "ice": {
                    "servers": self.args.ice_servers,
                    "transport_policy": self.args.ice_transport_policy,
                },
                "auth_enabled": self.args.enable_api_auth,
            },
        }
        payload["debug"] = {
            "enabled": bool(self.args.debug),
            "events_limit": self.debug_events_limit,
            "events": self.debug_events[-50:],
        }
        return web.json_response(payload)

    async def snapshot(self, _request: web.Request) -> web.Response:
        """Return latest frame as JPEG for quick visual checks."""

        data = self.video_buffer.snapshot_jpeg()
        if not data:
            return web.Response(status=503, text="snapshot unavailable")
        return web.Response(body=data, content_type="image/jpeg")

    async def healthz(self, _request: web.Request) -> web.Response:
        """Lightweight liveness/readiness endpoint."""

        await self._expire_stale_sessions()
        return web.json_response(
            {
                "ok": True,
                "aiortc_available": AIORTC_AVAILABLE,
                "uptime_seconds": round(time.time() - self.started_epoch, 1),
                "active_session_id": self.active_session_id,
                "session_count": len(self.sessions),
            }
        )

    async def config_v1(self, _request: web.Request) -> web.Response:
        """Return v1 API metadata including token header contract."""

        return web.json_response(
            {
                "rtc_config": self._browser_rtc_config(),
                "single_session_mode": self.args.single_session_mode,
                "session_token_header": SESSION_TOKEN_HEADER,
                "auth_enabled": self.args.enable_api_auth,
            }
        )

    async def active_session(self, _request: web.Request) -> web.Response:
        """Return current active session payload, if any."""

        await self._expire_stale_sessions()
        session = self._active_session()
        if session is None:
            return web.json_response({"active_session": None})
        return web.json_response({"active_session": self._session_payload(session)})

    def _active_session(self) -> Optional[SessionState]:
        """Resolve active session, repairing stale pointer when needed."""

        if self.active_session_id:
            session = self.sessions.get(self.active_session_id)
            if session is not None:
                return session
            self.active_session_id = None
        for session in self.sessions.values():
            state = self._session_state(session)
            if state not in {"closed", "failed"}:
                self.active_session_id = session.session_id
                return session
        return None

    def _create_session(self) -> SessionState:
        """Create and register a new in-memory session/token pair."""

        now = time.time()
        session = SessionState(
            session_id=uuid.uuid4().hex,
            token=secrets.token_urlsafe(24),
            created_epoch=now,
            last_activity_epoch=now,
        )
        self.sessions[session.session_id] = session
        self.active_session_id = session.session_id
        self._debug("session.created", session_id=session.session_id)
        return session

    def _get_session(self, session_id: str) -> Optional[SessionState]:
        """Lookup a session by id."""

        return self.sessions.get(session_id)

    def _session_conflict_payload(self, session: SessionState) -> dict:
        """Build standardized 409 payload for single-session conflicts."""

        return {
            "error": "single-session mode already has an active session",
            "active_session": self._session_payload(session),
            "hint": "Pass {\"replace\": true} to POST /v1/sessions to replace it.",
        }

    async def create_session(self, request: web.Request) -> web.Response:
        """Create session via `/v1/sessions` with optional replace semantics."""

        await self._expire_stale_sessions()
        body = await self._safe_json(request)
        replace = self._coerce_bool(body.get("replace"), default=False) or self._coerce_bool(
            request.query.get("replace"),
            default=False,
        )
        if self.args.single_session_mode:
            active = self._active_session()
            if active is not None:
                if not replace:
                    return web.json_response(self._session_conflict_payload(active), status=409)
                await self._close_session(active, reason="replaced")

        session = self._create_session()
        return web.json_response(
            {
                "session_id": session.session_id,
                "token": session.token,
                "created_epoch": session.created_epoch,
                "session_token_header": SESSION_TOKEN_HEADER,
                "single_session_mode": self.args.single_session_mode,
            }
        )

    def _check_session_token(self, request: web.Request, session: SessionState) -> Optional[web.Response]:
        """Validate session token header; return error response or `None`."""

        got = request.headers.get(SESSION_TOKEN_HEADER, "")
        if not got:
            return web.json_response({"error": f"missing {SESSION_TOKEN_HEADER} header"}, status=401)
        if not secrets.compare_digest(got, session.token):
            return web.json_response({"error": "invalid session token"}, status=401)
        return None

    async def delete_session(self, request: web.Request) -> web.Response:
        """Delete an existing session and close associated resources."""

        await self._expire_stale_sessions()
        sid = request.match_info["session_id"]
        session = self._get_session(sid)
        if session is None:
            return web.json_response({"error": "session not found"}, status=404)
        token_err = self._check_session_token(request, session)
        if token_err is not None:
            return token_err
        await self._close_session(session, reason="client_delete")
        return web.json_response({"ok": True, "session_id": sid})

    async def session_stats(self, request: web.Request) -> web.Response:
        """Return per-session counters/state for the requested session."""

        await self._expire_stale_sessions()
        sid = request.match_info["session_id"]
        session = self._get_session(sid)
        if session is None:
            return web.json_response({"error": "session not found"}, status=404)
        token_err = self._check_session_token(request, session)
        if token_err is not None:
            return token_err
        self._touch_session(session)
        return web.json_response(self._session_payload(session))

    def _session_payload(self, session: SessionState) -> dict:
        """Normalize session state into API response shape."""

        now = time.time()
        bridge = session.personaplex_bridge
        bridge_last_error = None
        bridge_uplink_q = 0
        bridge_handshake_epoch = None
        bridge_rx_packets = 0
        bridge_tx_packets = 0
        if bridge is not None:
            bridge_last_error = bridge.last_error or None
            bridge_uplink_q = int(bridge.uplink_queue.qsize())
            bridge_handshake_epoch = bridge.handshake_epoch or None
            bridge_rx_packets = int(bridge.rx_packets)
            bridge_tx_packets = int(bridge.tx_packets)
        return {
            "session_id": session.session_id,
            "created_epoch": session.created_epoch,
            "age_seconds": round(max(0.0, now - session.created_epoch), 3),
            "last_activity_epoch": session.last_activity_epoch,
            "last_offer_epoch": session.last_offer_epoch or None,
            "pc_state": self.pc_states.get(session.pcid or "", "new"),
            "mic_frames_rx": session.mic_frames_rx,
            "mic_samples_rx_16k": session.mic_samples_rx_16k,
            "last_mic_rx_epoch": session.last_mic_rx_epoch or None,
            "personaplex_connected": session.personaplex_connected,
            "personaplex_audio_frames_tx": session.personaplex_audio_frames_tx,
            "personaplex_audio_frames_rx": session.personaplex_audio_frames_rx,
            "personaplex_last_rx_epoch": session.personaplex_last_rx_epoch or None,
            "personaplex_bridge_last_error": bridge_last_error,
            "personaplex_uplink_queue_size": bridge_uplink_q,
            "personaplex_handshake_epoch": bridge_handshake_epoch,
            "personaplex_rx_packets": bridge_rx_packets,
            "personaplex_tx_packets": bridge_tx_packets,
            "active": session.session_id == self.active_session_id,
            "close_reason": session.close_reason or None,
        }

    async def _start_personaplex_chat_bridge(self, session: SessionState) -> None:
        """Start or restart per-session duplex PersonaPlex bridge task."""

        if not self._personaplex_chat_enabled():
            return
        self._debug("personaplex_chat.starting", session_id=session.session_id)
        if session.personaplex_bridge is not None:
            session.personaplex_bridge.stop_event.set()
        if session.personaplex_bridge_task is not None:
            session.personaplex_bridge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await session.personaplex_bridge_task
            session.personaplex_bridge_task = None

        ws_url = self._build_mirror_ws_url()
        bridge = PersonaPlexChatBridge(
            ws_url=ws_url,
            session=session,
            pcm_ring_24k=self.pcm_ring_24k,
            pcm_ring_16k=self.pcm_ring_16k,
            audio_track_buffer=self.audio_track_buffer,
            reconnect_delay_seconds=self.args.reconnect_delay_seconds,
            debug_log=self._debug,
        )
        session.personaplex_bridge = bridge
        session.personaplex_bridge_task = asyncio.create_task(bridge.run())
        self._debug("personaplex_chat.started", session_id=session.session_id, ws_url=ws_url)
        print(f"[personaplex-chat] started for session={session.session_id} url={ws_url}")

    async def offer(self, request: web.Request) -> web.Response:
        """Legacy offer endpoint that auto-creates session and returns answer."""

        await self._expire_stale_sessions()
        self._debug("offer.legacy.request", remote=request.remote or "")
        if self.args.single_session_mode:
            active = self._active_session()
            if active is not None:
                await self._close_session(active, reason="legacy_offer_replaced")
        session = self._create_session()
        return await self._handle_offer(request, session=session, include_session_token=True)

    async def session_offer(self, request: web.Request) -> web.Response:
        """Session-bound offer endpoint (`/v1/sessions/{id}/offer`)."""

        await self._expire_stale_sessions()
        sid = request.match_info["session_id"]
        self._debug("offer.session.request", remote=request.remote or "", session_id=sid)
        session = self._get_session(sid)
        if session is None:
            return web.json_response({"error": "session not found"}, status=404)
        token_err = self._check_session_token(request, session)
        if token_err is not None:
            return token_err
        if self.args.single_session_mode and self.active_session_id and self.active_session_id != sid:
            active = self._active_session()
            if active is not None:
                return web.json_response(self._session_conflict_payload(active), status=409)
        return await self._handle_offer(request, session=session)

    async def _handle_offer(
        self,
        request: web.Request,
        session: SessionState,
        include_session_token: bool = False,
    ) -> web.Response:
        """Validate SDP offer, bind tracks, and return SDP answer payload."""

        if not AIORTC_AVAILABLE:
            return web.json_response(
                {"error": "aiortc/av not installed. Install: pip install aiortc av"},
                status=500,
            )
        params = await self._safe_json(request)
        sdp = params.get("sdp")
        offer_type = str(params.get("type", "")).lower()
        self._debug(
            "offer.received",
            session_id=session.session_id,
            remote=request.remote or "",
            offer_type=offer_type,
            sdp_len=(len(sdp) if isinstance(sdp, str) else -1),
        )
        if not isinstance(sdp, str) or not sdp.strip() or offer_type != "offer":
            self._debug("offer.invalid_payload", session_id=session.session_id)
            return web.json_response({"error": "invalid offer payload; expected {type:'offer', sdp:'...'}"}, status=400)
        offer = RTCSessionDescription(sdp=sdp, type=offer_type)
        if session.pc is not None:
            with contextlib.suppress(Exception):
                await session.pc.close()
            self.pcs.discard(session.pc)

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=self._aiortc_ice_servers()))
        self.pcs.add(pc)
        pcid = f"pc_{id(pc)}"
        self.pc_states[pcid] = pc.connectionState
        session.pc = pc
        session.pcid = pcid
        session.last_offer_epoch = time.time()
        self._touch_session(session)
        self.active_session_id = session.session_id
        self._debug("offer.pc_created", session_id=session.session_id, pcid=pcid)
        await self._start_personaplex_chat_bridge(session)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            """Track peer state transitions and cleanup failed/closed peers."""

            self.pc_states[pcid] = pc.connectionState
            self._touch_session(session)
            self._debug(
                "webrtc.connection_state",
                session_id=session.session_id,
                pcid=pcid,
                state=pc.connectionState,
            )
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self.pcs.discard(pc)
                self.pc_states[pcid] = "closed"
                if session.inbound_task is not None:
                    session.inbound_task.cancel()
                    session.inbound_task = None

        @pc.on("track")
        async def on_track(track):
            """Attach inbound browser mic audio consumer when audio track arrives."""

            self._debug(
                "webrtc.track",
                session_id=session.session_id,
                kind=getattr(track, "kind", "unknown"),
            )
            if getattr(track, "kind", "") != "audio":
                return
            if session.inbound_task is not None:
                session.inbound_task.cancel()
            session.inbound_task = asyncio.create_task(self._consume_inbound_audio(track, session))
            self._touch_session(session)

        video_track = MuseTalkVideoTrack(self.video_buffer, fps=self.args.fps, stats=self.track_stats)
        audio_track = MuseTalkAudioTrack(self.audio_track_buffer, stats=self.track_stats)
        pc.addTrack(video_track)
        pc.addTrack(audio_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self._wait_for_ice_gathering(pc, timeout=3.0)
        self._debug("offer.answer_ready", session_id=session.session_id, pcid=pcid)

        payload = {
            "session_id": session.session_id,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
        if include_session_token:
            payload["session_token"] = session.token
            payload["session_token_header"] = SESSION_TOKEN_HEADER
        return web.json_response(payload)

    async def _consume_inbound_audio(self, track, session: SessionState) -> None:
        """Read browser mic frames, update stats, and route audio by mode.

        Receives:
        - `track`: inbound aiortc audio track from browser mic.
        - `session`: owner session state.

        Returns:
        - `None` (infinite loop until track closes/cancelled).
        """

        frame_counter = 0
        try:
            while True:
                frame = await track.recv()
                pcm = frame.to_ndarray()
                pcm = np.asarray(pcm)
                channels = len(frame.layout.channels)
                if channels > 1:
                    if getattr(frame.format, "is_planar", False):
                        pcm = pcm.mean(axis=0)
                    else:
                        pcm = pcm.reshape(-1, channels).mean(axis=1)

                pcm = pcm.reshape(-1).astype(np.float32, copy=False)
                fmt_name = str(getattr(getattr(frame, "format", None), "name", "")).lower()
                if fmt_name.startswith(("s16", "s32", "u8")):
                    pcm = pcm / 32768.0
                src_sr = int(getattr(frame, "sample_rate", 48000) or 48000)
                if pcm.size == 0:
                    continue

                pcm16k = self._resample_audio(pcm, src_sr, 16000)
                session.mic_samples_rx_16k += int(pcm16k.size)
                session.mic_frames_rx += 1
                session.last_mic_rx_epoch = time.time()
                self._touch_session(session)
                frame_counter += 1
                if frame_counter in {1, 50}:
                    self._debug(
                        "mic.rx",
                        session_id=session.session_id,
                        frames=int(session.mic_frames_rx),
                        src_sr=src_sr,
                        samples=int(pcm.size),
                    )

                if session.personaplex_bridge is not None:
                    pcm24k_uplink = self._resample_audio(pcm, src_sr, 24000)
                    await session.personaplex_bridge.push_uplink_pcm24k(pcm24k_uplink)

                if self.args.input_source in ("webrtc", "mixed") and session.personaplex_bridge is None:
                    await self.pcm_ring_16k.append(pcm16k)

                if self.args.webrtc_audio_loopback and session.personaplex_bridge is None:
                    pcm24k = self._resample_audio(pcm, src_sr, 24000)
                    await self.pcm_ring_24k.append(pcm24k)
                    await self.audio_track_buffer.append_from_24k(pcm24k)
        except asyncio.CancelledError:
            self._debug("mic.loop_cancelled", session_id=session.session_id)
            raise
        except Exception as e:
            self._debug("mic.loop_error", session_id=session.session_id, error=repr(e))
            raise

    def _resample_audio(self, pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Resample mono PCM to target sample rate using polyphase filter."""

        if src_sr == dst_sr:
            return pcm.astype(np.float32, copy=False)
        if src_sr <= 0 or dst_sr <= 0:
            return np.zeros((0,), dtype=np.float32)
        return resample_poly(pcm, up=dst_sr, down=src_sr).astype(np.float32, copy=False)

    @web.middleware
    async def auth_middleware(self, request: web.Request, handler):
        """Optional bearer auth middleware for `/v1/*` endpoints."""

        if not self.args.enable_api_auth:
            return await handler(request)
        if not request.path.startswith("/v1/"):
            return await handler(request)
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {self.args.api_token}"
        if not self.args.api_token or auth != expected:
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    async def _wait_for_ice_gathering(self, pc: RTCPeerConnection, timeout: float) -> None:
        """Wait until ICE gathering completes, or until timeout elapses."""

        if pc.iceGatheringState == "complete":
            return
        done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        async def _on_ice_state_change():
            """Signal waiter once ICE gathering reaches `complete`."""

            if pc.iceGatheringState == "complete":
                done.set()

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(done.wait(), timeout=timeout)
        if pc.iceGatheringState != "complete":
            self._debug("webrtc.ice_gather_timeout", timeout=timeout, state=pc.iceGatheringState)

    def build_app(self) -> web.Application:
        """Create aiohttp app and register all public routes."""

        app = web.Application(middlewares=[self.auth_middleware])
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        app.router.add_get("/", self.index)
        app.router.add_get("/healthz", self.healthz)
        app.router.add_get("/config", self.config)
        app.router.add_get("/status", self.status)
        app.router.add_get("/snapshot.jpg", self.snapshot)
        app.router.add_post("/offer", self.offer)
        app.router.add_get("/v1/config", self.config_v1)
        app.router.add_get("/v1/session", self.active_session)
        app.router.add_post("/v1/sessions", self.create_session)
        app.router.add_post("/v1/sessions/{session_id}/offer", self.session_offer)
        app.router.add_get("/v1/sessions/{session_id}/stats", self.session_stats)
        app.router.add_delete("/v1/sessions/{session_id}", self.delete_session)
        return app
