# In-Memory PersonaPlex -> MuseTalk -> WebRTC

This path removes the slow disk loop:
- No rolling WAV files
- No per-chunk MP4 encode/decode
- MuseTalk stays resident in memory
- Browser receives frames/audio by WebRTC

## Install additional dependencies

```bash
pip install aiortc av
```

## Run

```bash
cd /teamspace/studios/this_studio/MuseTalk
python -m scripts.personaplex_musetalk_webrtc \
  --avatar-id my_avatar \
  --version v15 \
  --gpu-id 0 \
  --use-fp16 \
  --personaplex-host 127.0.0.1 \
  --personaplex-port 8998 \
  --personaplex-path /api/avatar/audio \
  --fps 20 \
  --batch-size 8 \
  --window-ms 400 \
  --hop-ms 60 \
  --min-window-ms 240 \
  --max-advance-ms 180 \
  --max-tail-frames 3 \
  --mouth-smoothing-alpha 0.7
```

Open:
- `http://<host>:8780/` (preview client)
- `http://<host>:8780/status` (diagnostics)

## MVP cloud signaling API (for native clients)

In addition to legacy `POST /offer`, the server now exposes session-based endpoints:

- `POST /v1/sessions` -> create a session and return `{session_id, token}`
- `POST /v1/sessions/{session_id}/offer` -> submit SDP offer and receive SDP answer
- `GET /v1/sessions/{session_id}/stats` -> per-session media counters
- `DELETE /v1/sessions/{session_id}` -> teardown

Useful flags for MVP duplex bring-up:

- `--input-source mirror|webrtc|mixed` (default: `mirror`)
- `--webrtc-audio-loopback` (echo inbound WebRTC mic to outbound audio track for testing)
- `--enable-api-auth` + `--api-token` (auth scaffold; off by default)

## Fixing "WebRTC stuck at connecting" on cloud

When server and browser are on different networks (NAT/proxy), host ICE candidates are often unreachable.
Use TURN/STUN and pass the same ICE servers to both peers via CLI:

```bash
python -m scripts.personaplex_musetalk_webrtc \
  ... \
  --ice-server "stun:stun.l.google.com:19302" \
  --ice-server "turn:YOUR_TURN_HOST:3478?transport=tcp" \
  --ice-server "turns:YOUR_TURN_HOST:5349?transport=tcp" \
  --ice-username "YOUR_TURN_USERNAME" \
  --ice-credential "YOUR_TURN_PASSWORD"
```

If your network blocks UDP, force relay:

```bash
python -m scripts.personaplex_musetalk_webrtc \
  ... \
  --ice-server "turn:YOUR_TURN_HOST:3478?transport=tcp" \
  --ice-username "YOUR_TURN_USERNAME" \
  --ice-credential "YOUR_TURN_PASSWORD" \
  --ice-transport-policy relay
```

## Repeated lips / loop fix

This pipeline explicitly publishes only **new tail frames** per hop from each overlapping audio window,
instead of replaying the entire window each cycle. That fixes the "same mouth movement repeated" effect
you saw in the disk-chunk workflow.

## Quality / artifact notes

- Best quality requires mmpose/DWPose face alignment (`--require-mmpose` once your env supports it).
- Headset boom mics near jawline can cause blend artifacts; use a prep video without mouth occlusion if possible.
- If motion looks stuck, lower `--fps` to `15` and keep `--hop-ms` around `50-70`.
