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
  --personaplex-host 127.0.0.1 \
  --personaplex-port 8998 \
  --personaplex-path /api/avatar/audio \
  --fps 25 \
  --batch-size 16 \
  --window-ms 640 \
  --hop-ms 80
```

Open:
- `http://<host>:8780/` (preview client)
- `http://<host>:8780/status` (diagnostics)

## Repeated lips / loop fix

This pipeline explicitly publishes only **new tail frames** per hop from each overlapping audio window,
instead of replaying the entire window each cycle. That fixes the "same mouth movement repeated" effect
you saw in the disk-chunk workflow.
