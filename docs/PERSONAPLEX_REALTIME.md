# PersonaPlex -> MuseTalk Realtime (Prepared Avatar)

This setup runs in two loops:
- `personaplex_audio_bridge.py`: PersonaPlex websocket audio -> rolling wav chunks.
- `musetalk_chunk_worker.py`: consumes latest rolling chunk and runs MuseTalk with your prepared avatar.

## 1) Start PersonaPlex server (Terminal A)

Run your PersonaPlex server normally (example):

```bash
cd /teamspace/studios/this_studio/personaplex
python -m moshi.moshi.server --host 0.0.0.0 --port 8998
```

This repo now exposes an additional websocket mirror endpoint:
- `/api/avatar/audio` (read-only live outbound audio packets from active calls)

If your server requires a voice prompt filename, note it (example: `s0.wav`).

## 2) Start MuseTalk chunk worker (Terminal B)

```bash
cd /teamspace/studios/this_studio/MuseTalk
python -m scripts.musetalk_chunk_worker \
  --avatar-id my_avatar \
  --version v15 \
  --gpu-id 0 \
  --unet-model-path models/musetalkV15/unet.pth \
  --unet-config models/musetalkV15/musetalk.json \
  --fps 25 \
  --batch-size 8 \
  --output-prefix live \
  --chunk-wav data/audio/live/latest_chunk.wav
```

Optional quality/speed flags:
- Add `--use-fp16` for speed.
- Omit `--use-fp16` for fp32 quality mode.
- Add `--require-mmpose` to fail fast if DWPose/mmpose is missing.

## 3) Start bridge (Terminal C)

### Mode A: Mirror a live default web call (recommended)

This mode does not replace `/api/chat`. It listens to the new mirror endpoint:
`/api/avatar/audio`.

```bash
cd /teamspace/studios/this_studio/MuseTalk
python -m scripts.personaplex_audio_bridge \
  --host 127.0.0.1 \
  --port 8998 \
  --path /api/avatar/audio \
  --chunk-path data/audio/live/latest_chunk.wav \
  --reply-wav data/audio/live/personaplex_reply.wav
```

Now use PersonaPlex web UI normally. As the call runs, MuseTalk worker will consume live mirrored audio chunks.

### Mode B: Standalone audio test (query-driven /api/chat)

Example with your `audio.wav` test input:

```bash
cd /teamspace/studios/this_studio/MuseTalk
python -m scripts.personaplex_audio_bridge \
  --host 127.0.0.1 \
  --port 8998 \
  --path /api/chat \
  --text-prompt "You are concise and friendly." \
  --voice-prompt s0.wav \
  --input-wav /teamspace/studios/this_studio/audio.wav \
  --chunk-path data/audio/live/latest_chunk.wav \
  --reply-wav data/audio/live/personaplex_reply.wav \
  --chunk-seconds 0.8 \
  --chunk-hop-ms 200 \
  --verbose-text
```

Notes:
- If your server already has full query params, use `--ws-url "ws://.../api/chat?..."`
- If your server does not enforce voice prompts, `--voice-prompt` can be empty.

## Outputs

- Rolling input chunk for MuseTalk:
  - `data/audio/live/latest_chunk.wav`
- Full PersonaPlex decoded reply audio:
  - `data/audio/live/personaplex_reply.wav`
- MuseTalk generated clips:
  - `results/v15/avatars/<avatar-id>/vid_output/live_000000.mp4`, `live_000001.mp4`, ...
- Latest still frame for streaming:
  - `data/video/live/latest.jpg`

## 4) Start MJPEG video stream (Terminal D)

```bash
cd /teamspace/studios/this_studio/MuseTalk
python -m scripts.avatar_mjpeg_server \
  --host 0.0.0.0 \
  --port 8767 \
  --latest-jpeg data/video/live/latest.jpg
```

Open:
- `http://<host>:8767/` (preview page)
- `http://<host>:8767/mjpeg` (raw stream URL)

If you want this inside your web app, embed:

```html
<img src="http://<host>:8767/mjpeg" />
```

### Optional: show stream directly inside PersonaPlex React UI

The client now supports `VITE_AVATAR_STREAM_URL`.

Set:

```bash
export VITE_AVATAR_STREAM_URL="http://127.0.0.1:8767/mjpeg"
```

Then run/build the PersonaPlex client as usual. Conversation page will render the avatar stream panel above audio controls.

## First run sanity check

1. Confirm avatar is already prepared (`preparation=True` step done earlier).
2. Start Terminal B (worker), then Terminal C (bridge).
3. Verify `latest_chunk.wav` updates over time.
4. Verify `live_*.mp4` appears under the avatar `vid_output` folder.
