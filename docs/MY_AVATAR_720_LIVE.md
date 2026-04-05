# My Avatar 720p25 Live Profile

This adds a dedicated profile for your validated avatar:
- video: `/teamspace/studios/this_studio/avatar_720p25.mp4`
- audio sanity clip: `/teamspace/studios/this_studio/audio.wav`
- avatar id: `my_avatar_720_live`

No existing PersonaPlex/MuseTalk runtime files were modified.

## 1) Prepare avatar once (GPU)

```bash
cd /teamspace/studios/this_studio/MuseTalk
chmod +x scripts/run_my_avatar_720_live_prepare.sh
./scripts/run_my_avatar_720_live_prepare.sh
```

Optional overrides:

```bash
GPU_ID=0 BATCH_SIZE=8 FPS=25 ./scripts/run_my_avatar_720_live_prepare.sh
```

## 2) Verify prepared cache exists

```bash
cd /teamspace/studios/this_studio/MuseTalk
ls -lh results/v15/avatars/my_avatar_720_live/latents.pt
ls -lh results/v15/avatars/my_avatar_720_live/coords.pkl
ls -lh results/v15/avatars/my_avatar_720_live/mask_coords.pkl
```

## 3) Run PersonaPlex + MuseTalk WebRTC (later, when ready)

Start PersonaPlex in its own terminal first.

Use the local source launcher (required for `/api/avatar/audio` mirror endpoint):

```bash
cd /teamspace/studios/this_studio/personaplex
./run_personaplex_with_avatar_mirror.sh
```

Then:

```bash
cd /teamspace/studios/this_studio/MuseTalk
chmod +x scripts/run_my_avatar_720_live_webrtc.sh
./scripts/run_my_avatar_720_live_webrtc.sh
```

If WebRTC stays in `connecting` on cloud, run with TURN relay:

```bash
ICE_SERVER="turn:YOUR_TURN_HOST:3478?transport=tcp" \
ICE_SERVER_2="turns:YOUR_TURN_HOST:5349?transport=tcp" \
ICE_USERNAME="YOUR_TURN_USERNAME" \
ICE_CREDENTIAL="YOUR_TURN_PASSWORD" \
ICE_TRANSPORT_POLICY=relay \
./scripts/run_my_avatar_720_live_webrtc.sh
```

If native TURN is already configured on this machine, use:

```bash
cd /teamspace/studios/this_studio/MuseTalk
./scripts/run_my_avatar_720_live_webrtc_with_turn.sh
```

Open:
- `http://<host>:8780/`
- `http://<host>:8780/status`

## Added files

- `configs/inference/runtime/my_avatar_720_live_step1.yaml`
- `configs/inference/runtime/my_avatar_720_live_reuse.yaml`
- `scripts/run_my_avatar_720_live_prepare.sh`
- `scripts/run_my_avatar_720_live_webrtc.sh`
