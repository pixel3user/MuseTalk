"""MuseTalk realtime inference engine wrapper."""

import argparse
import asyncio
import math
import os
import shutil
import time

import cv2
import numpy as np
import torch
from einops import rearrange

from .buffers import PcmRingBuffer, VideoFrameBuffer
from .models import AppArgs

async def _frame_gap_watchdog(video_buffer, fps, stats):
    """Logs when no frame has been published for more than 2 frame intervals."""
    frame_interval = 1.0 / fps
    threshold = frame_interval * 2  # 80ms at 25fps
    while True:
        await asyncio.sleep(frame_interval)
        last = stats.get("last_video_send_epoch", 0)
        gap = time.time() - last
        if last > 0 and gap > threshold:
            q = video_buffer.queue.qsize()
            print(f"[WATCHDOG] *** NO FRAME FOR {gap*1000:.0f}ms *** queue={q}")

class MuseTalkRealtimeEngine:
    """Audio-driven MuseTalk inference engine that publishes avatar frames."""

    def __init__(
        self,
        args: AppArgs,
        pcm_ring_16k: PcmRingBuffer,
        video_buffer: VideoFrameBuffer,
    ):
        """Initialize heavy MuseTalk realtime inference runtime.

        Receives:
        - `args`: model/runtime configuration.
        - `pcm_ring_16k`: audio input ring (driving lip-sync inference).
        - `video_buffer`: output frame queue for WebRTC video track.

        Returns:
        - `None`.
        """

        self.args = args
        self.pcm_ring = pcm_ring_16k
        self.video_buffer = video_buffer
        self.stop_event = asyncio.Event()
        self.last_total_samples = -1
        self.avatar_time_secs = 0.0
        self.jobs = 0
        self.dropped_audio_ms_total = 0.0
        self.last_publish_epoch = 0.0
        self.last_error = ""
        self.prev_mouth_patch = None
        self._whisper_feature_cache = None
        self._cache_audio_samples = 0
        self.vad_hangover = 0
        self.silero_vad_model = None
        self.silero_get_timestamps = None
        
        # Performance Tracking (Claude Recommendation)
        self._infer_ms_ema = 320.0 # Initial estimate to match observed latency
        self.stats = {
            "last_video_send_epoch": 0.0
        }

        import scripts.realtime_inference as rt

        self.rt = rt
        self._setup_runtime()
        self.avatar = self.rt.Avatar(
            avatar_id=args.avatar_id,
            video_path="unused",
            bbox_shift=args.bbox_shift,
            batch_size=args.batch_size,
            preparation=False,
        )
        # Seed a non-black idle frame so preview is visible before first audio packets arrive.
        if getattr(self.avatar, "frame_list_cycle", None):
            try:
                self.video_buffer.last_frame = self.avatar.frame_list_cycle[0].copy()
            except Exception:
                pass

    def _setup_runtime(self) -> None:
        """Load and configure MuseTalk/Whisper/FaceParsing runtime objects.

        Receives:
        - None (uses `self.args`).

        Returns:
        - `None`.
        """

        rt = self.rt
        rt.args = argparse.Namespace(
            version=self.args.version,
            ffmpeg_path=self.args.ffmpeg_path,
            gpu_id=self.args.gpu_id,
            vae_type=self.args.vae_type,
            unet_config=self.args.unet_config,
            unet_model_path=self.args.unet_model_path,
            whisper_dir=self.args.whisper_dir,
            inference_config="",
            bbox_shift=self.args.bbox_shift,
            result_dir="results",
            extra_margin=self.args.extra_margin,
            fps=self.args.fps,
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
            batch_size=self.args.batch_size,
            output_vid_name=None,
            use_saved_coord=False,
            saved_coord=False,
            parsing_mode=self.args.parsing_mode,
            left_cheek_width=self.args.left_cheek_width,
            right_cheek_width=self.args.right_cheek_width,
            skip_save_images=True,
            non_interactive=True,
            force_recreate_avatar=False,
            use_fp16=self.args.use_fp16,
            require_mmpose=self.args.require_mmpose,
        )
        rt.device = torch.device(f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
        if rt.device.type != "cuda":
            print("[engine][warn] CUDA is unavailable; realtime inference will be very slow on CPU.")
        if not shutil.which("ffmpeg"):
            print("[engine][warn] ffmpeg not found in PATH (not required for core realtime generation).")

        if rt.args.require_mmpose and not rt.MMPOSE_AVAILABLE:
            raise RuntimeError("mmpose/DWPose is required but unavailable.")
        if not rt.MMPOSE_AVAILABLE:
            print("[engine][warn] mmpose unavailable; using fallback face detector.")
        vae_config = os.path.join("models", rt.args.vae_type, "config.json")
        if not os.path.exists(vae_config):
            raise FileNotFoundError(
                f"Missing VAE config: {vae_config}. Download weights with ./download_weights.sh "
                "or set --vae-type to a valid local VAE directory under ./models."
            )

        rt.vae, rt.unet, rt.pe = rt.load_all_model(
            unet_model_path=rt.args.unet_model_path,
            vae_type=rt.args.vae_type,
            unet_config=rt.args.unet_config,
            device=rt.device,
        )
        rt.timesteps = torch.tensor([0], device=rt.device)

        if rt.device.type == "cuda" and rt.args.use_fp16:
            # Optimal precision choice for Ada/Blackwell GPUs
            if torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
                precision_name = "bf16"
            else:
                target_dtype = torch.float16
                precision_name = "fp16"
            
            rt.pe = rt.pe.to(device=rt.device, dtype=target_dtype)
            rt.vae.vae = rt.vae.vae.to(device=rt.device, dtype=target_dtype)
            rt.unet.model = rt.unet.model.to(device=rt.device, dtype=target_dtype)
            
            # Enable hardware-accelerated Flash Attention if available
            try:
                rt.unet.model.enable_xformers_memory_efficient_attention()
            except Exception:
                # Fallback to PyTorch native SDPA which is also very fast
                pass
                
            print(f"[engine] precision: {precision_name} (with attention optimization)")
        else:
            rt.pe = rt.pe.float().to(rt.device)
            rt.vae.vae = rt.vae.vae.float().to(rt.device)
            rt.unet.model = rt.unet.model.float().to(rt.device)
            print("[engine] precision: fp32")
        rt.pe.eval()
        rt.vae.vae.eval()
        rt.unet.model.eval()
        rt.pe.requires_grad_(False)
        rt.vae.vae.requires_grad_(False)
        rt.unet.model.requires_grad_(False)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        rt.audio_processor = rt.AudioProcessor(feature_extractor_path=rt.args.whisper_dir)
        rt.weight_dtype = rt.unet.model.dtype
        rt.whisper = rt.WhisperModel.from_pretrained(rt.args.whisper_dir)
        rt.whisper = rt.whisper.to(device=rt.device, dtype=rt.weight_dtype).eval()
        rt.whisper.requires_grad_(False)

        if rt.args.version == "v15":
            rt.fp = rt.FaceParsing(
                left_cheek_width=rt.args.left_cheek_width,
                right_cheek_width=rt.args.right_cheek_width,
            )
        else:
            rt.fp = rt.FaceParsing()

        print("[engine] Loading Silero VAD...")
        self.silero_vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True
        )
        self.silero_get_timestamps = utils[0]
        self.silero_vad_model = self.silero_vad_model.to(rt.device)
        self.silero_vad_model.eval()

    @torch.no_grad()
    def _append_whisper_cache(self, new_pcm16k: np.ndarray) -> None:
        """Incrementally append Whisper hidden features for only the newest audio."""

        if new_pcm16k.size == 0:
            return
        rt = self.rt
        feature_ret = rt.audio_processor.get_audio_feature_from_array(
            new_pcm16k, sample_rate=16000, weight_dtype=rt.weight_dtype
        )
        if feature_ret is None:
            return
        whisper_input_features, _ = feature_ret
        new_hidden = []
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(rt.device).to(rt.weight_dtype)
            audio_feats = rt.whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            audio_feats = torch.stack(audio_feats, dim=2)
            new_hidden.append(audio_feats)
        if not new_hidden:
            return
        appended = torch.cat(new_hidden, dim=1)
        if self._whisper_feature_cache is None:
            self._whisper_feature_cache = appended
        else:
            self._whisper_feature_cache = torch.cat([self._whisper_feature_cache, appended], dim=1)
        self._cache_audio_samples += int(new_pcm16k.size)

        # Keep only a bounded recent cache (window + safety margin).
        keep_ms = max(
            int(self.args.window_ms + self.args.min_window_ms + self.args.max_advance_ms + 200),
            1000,
        )
        keep_samples = int((keep_ms / 1000.0) * 16000)
        if self._cache_audio_samples <= keep_samples:
            return
        drop_samples = self._cache_audio_samples - keep_samples
        drop_frames = max(0, int(math.floor((drop_samples / 16000.0) * 50.0)))
        if drop_frames > 0 and self._whisper_feature_cache is not None:
            self._whisper_feature_cache = self._whisper_feature_cache[:, drop_frames:, ...]
        self._cache_audio_samples = keep_samples

    @torch.no_grad()
    def _tail_whisper_chunks(self, new_frames: int) -> torch.Tensor | None:
        """Build only tail Whisper chunks from cached hidden states."""

        if self._whisper_feature_cache is None:
            return None
        rt = self.rt
        sr = 16000
        audio_fps = 50
        fps = int(self.args.fps)
        if fps <= 0:
            return None
        whisper_idx_multiplier = audio_fps / fps
        num_frames_total = math.floor((self._cache_audio_samples / sr) * fps)
        if num_frames_total <= 0:
            return None
        tail = max(1, min(int(new_frames), int(self.args.max_tail_frames), num_frames_total))
        start_frame = max(0, num_frames_total - tail)
        audio_feature_length_per_frame = 2 * (
            self.args.audio_padding_length_left + self.args.audio_padding_length_right + 1
        )

        actual_length = min(
            self._whisper_feature_cache.shape[1],
            max(0, math.floor((self._cache_audio_samples / sr) * audio_fps)),
        )
        whisper_feature = self._whisper_feature_cache[:, :actual_length, ...]
        if whisper_feature.shape[1] <= 0:
            return None

        padding_nums = math.ceil(whisper_idx_multiplier)
        whisper_feature = torch.cat(
            [
                torch.zeros_like(whisper_feature[:, : padding_nums * self.args.audio_padding_length_left]),
                whisper_feature,
                torch.zeros_like(whisper_feature[:, : padding_nums * 3 * self.args.audio_padding_length_right]),
            ],
            1,
        )

        audio_prompts = []
        for frame_index in range(start_frame, num_frames_total):
            audio_index = math.floor(frame_index * whisper_idx_multiplier)
            audio_clip = whisper_feature[:, audio_index : audio_index + audio_feature_length_per_frame]
            if audio_clip.shape[1] != audio_feature_length_per_frame:
                continue
            audio_prompts.append(audio_clip)
        if not audio_prompts:
            return None
        audio_prompts = torch.cat(audio_prompts, dim=0)
        audio_prompts = rearrange(audio_prompts, "b c h w -> b (c h) w")
        return audio_prompts

    @torch.no_grad()
    def _infer_window_frames(
        self,
        pcm16k_window: np.ndarray,
        new_frames: int,
        advance_samples: int,
    ) -> list[np.ndarray]:
        """Run one inference pass and return only the newest tail frames.

        Receives:
        - `pcm16k_window`: latest inference window at 16kHz.
        - `new_frames`: number of most-recent frames to publish.

        Returns:
        - List of BGR frames (can be empty when features are unavailable).
        """

        if advance_samples <= 0:
            return []
        tail_n = max(1, min(int(advance_samples), int(pcm16k_window.size)))
        self._append_whisper_cache(pcm16k_window[-tail_n:])
        whisper_chunks = self._tail_whisper_chunks(new_frames)
        
        # If cache doesn't have enough chunks, re-encode the full window (Claude Recommendation)
        if whisper_chunks is None or len(whisper_chunks) < new_frames:
            self._whisper_feature_cache = None
            self._cache_audio_samples = 0
            # Feed the full window (e.g. 640ms) to ensure enough chunks exist for the batch
            self._append_whisper_cache(pcm16k_window)
            whisper_chunks = self._tail_whisper_chunks(new_frames)

        if whisper_chunks is None or len(whisper_chunks) == 0:
            return []

        rt = self.rt
        combined_frames = []

        def my_datagen(whisper_chunks, vae_encode_latents, batch_size, start_time_secs, fps, avatar_fps):
            whisper_batch, latent_batch = [], []
            for i, w in enumerate(whisper_chunks):
                t = start_time_secs + i / float(fps)
                idx = int(t * avatar_fps) % len(vae_encode_latents)
                latent_batch.append(vae_encode_latents[idx])
                whisper_batch.append(w)
                if len(latent_batch) >= batch_size:
                    yield torch.stack(whisper_batch), torch.cat(latent_batch, dim=0)
                    whisper_batch, latent_batch = [], []
            if len(latent_batch) > 0:
                yield torch.stack(whisper_batch), torch.cat(latent_batch, dim=0)

        gen = my_datagen(
            whisper_chunks,
            self.avatar.input_latent_list_cycle,
            self.args.batch_size,
            self.avatar_time_secs,
            self.args.fps,
            self.args.avatar_fps
        )

        t0 = time.time()
        model_time = 0.0
        cv_time = 0.0

        for whisper_batch, latent_batch in gen:
            t_model_start = time.time()
            audio_feature_batch = rt.pe(whisper_batch.to(rt.device))
            latent_batch = latent_batch.to(device=rt.device, dtype=rt.unet.model.dtype)
            pred_latents = rt.unet.model(
                latent_batch, rt.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents = pred_latents.to(device=rt.device, dtype=rt.vae.vae.dtype)
            recon = rt.vae.decode_latents(pred_latents)
            # Ensure GPU finishes before measuring
            if rt.device.type == "cuda":
                torch.cuda.synchronize()
            model_time += time.time() - t_model_start

            t_cv_start = time.time()
            for res_frame in recon:
                base_i = int(self.avatar_time_secs * self.args.avatar_fps) % len(self.avatar.frame_list_cycle)
                bbox = self.avatar.coord_list_cycle[base_i]
                ori_frame = self.avatar.frame_list_cycle[base_i]
                x1, y1, x2, y2 = bbox
                try:
                    lip = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    self.avatar_time_secs += 1.0 / self.args.fps
                    continue
                mask = self.avatar.mask_list_cycle[base_i]
                mask_box = self.avatar.mask_coords_list_cycle[base_i]
                frame = rt.get_image_blending(ori_frame, lip, bbox, mask, mask_box)
                # Temporal smoothing on the mouth patch reduces flicker/chin artifacts.
                if 0.0 <= self.args.mouth_smoothing_alpha < 1.0:
                    x1c = max(0, min(frame.shape[1] - 1, x1))
                    x2c = max(1, min(frame.shape[1], x2))
                    y1c = max(0, min(frame.shape[0] - 1, y1))
                    y2c = max(1, min(frame.shape[0], y2))
                    cur = frame[y1c:y2c, x1c:x2c]
                    if self.prev_mouth_patch is not None and self.prev_mouth_patch.shape == cur.shape:
                        alpha = float(self.args.mouth_smoothing_alpha)
                        cur = cv2.addWeighted(cur, alpha, self.prev_mouth_patch, 1.0 - alpha, 0.0)
                        frame[y1c:y2c, x1c:x2c] = cur
                    self.prev_mouth_patch = frame[y1c:y2c, x1c:x2c].copy()
                combined_frames.append(frame)
                self.avatar_time_secs += 1.0 / self.args.fps
            cv_time += time.time() - t_cv_start

        if self.args.debug and new_frames > 0:
            print(f"[profiler] frames={new_frames} model={model_time*1000:.1f}ms cv={cv_time*1000:.1f}ms total={(time.time()-t0)*1000:.1f}ms")

        if not combined_frames:
            return []
        if new_frames <= 0:
            return [combined_frames[-1]]
        return combined_frames[-new_frames:]

    async def run(self) -> None:
        """Main loop that advances inference from the rolling 16k audio ring.

        Receives:
        - None.

        Returns:
        - `None` (runs until `stop_event` is set).
        """

        # Start watchdog within the active event loop (Claude Recommendation)
        asyncio.create_task(_frame_gap_watchdog(self.video_buffer, self.args.fps, self.stats))

        window_samples = int((self.args.window_ms / 1000.0) * 16000)
        min_samples = int((self.args.min_window_ms / 1000.0) * 16000)
        max_advance_samples = int((self.args.max_advance_ms / 1000.0) * 16000)

        min_advance_samples = int((self.args.hop_ms / 1000.0) * 16000)

        while not self.stop_event.is_set():
            # Event-driven wait for new audio data
            _loop_start = time.perf_counter()
            window, total = await self.pcm_ring.latest(window_samples)
            
            if self.last_total_samples < 0:
                new_samples = int(window.size)
            else:
                new_samples = max(0, int(total - self.last_total_samples))

            # If no new samples or not enough for a hop, wait for the event
            if new_samples < min_advance_samples:
                try:
                    # Wait for up to 100ms for new data
                    await asyncio.wait_for(self.pcm_ring.new_data_event.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
                self.pcm_ring.new_data_event.clear()
                continue

            self.last_total_samples = total

            if max_advance_samples > 0 and new_samples > max_advance_samples:
                dropped = (new_samples - max_advance_samples) * 1000.0 / 16000.0
                self.dropped_audio_ms_total += dropped
                new_samples = max_advance_samples

            # --- VAD Filtering (Offloaded to background thread to avoid event loop stalls) ---
            new_audio = window[-new_samples:]
            
            def _run_vad(audio_chunk):
                min_samples_silero = 512
                vad_audio_chunk = audio_chunk
                if len(vad_audio_chunk) < min_samples_silero:
                    vad_audio_chunk = np.pad(vad_audio_chunk, (0, min_samples_silero - len(vad_audio_chunk)))

                tensor_audio = torch.from_numpy(vad_audio_chunk).to(self.rt.device)
                return self.silero_get_timestamps(
                    tensor_audio,
                    self.silero_vad_model,
                    sampling_rate=16000,
                    min_speech_duration_ms=20,
                    min_silence_duration_ms=10
                )

            speech_timestamps = await asyncio.to_thread(_run_vad, new_audio)
            
            _vad_ms = (time.perf_counter() - _loop_start) * 1000
            if _vad_ms > 10:
                print(f"[ENGINE] VAD processing took {_vad_ms:.1f}ms (running in background thread)")

            speech_detected = len(speech_timestamps) > 0

            if speech_detected:
                self.vad_hangover = 15
            else:
                self.vad_hangover = max(0, self.vad_hangover - 1)

            if self.vad_hangover == 0:
                # No speech detected recently. Zero out the newest samples.
                window[-new_samples:] = 0.0
                # Flush the whisper cache to prevent mouth "stuttering" on old speech features
                self._whisper_feature_cache = None
                self._cache_audio_samples = 0

            # Adaptive frame production to cover inference latency (Claude Recommendation)
            # Calculate how many frames we need to generate to cover the time spent in the loop
            frames_to_cover_latency = math.ceil(self._infer_ms_ema / (1000.0 / self.args.fps))
            
            # Base frame count from audio advancement
            new_frames = max(1, int(round((new_samples / 16000.0) * self.args.fps)))
            
            # Use the higher of the two, capped at batch size to stay real-time
            new_frames = max(new_frames, frames_to_cover_latency)
            new_frames = min(new_frames, self.args.batch_size)

            try:
                frames = await asyncio.to_thread(self._infer_window_frames, window, new_frames, new_samples)
                
                _infer_ms = (time.perf_counter() - _loop_start) * 1000
                print(f"[ENGINE] infer done: {len(frames)} frames in {_infer_ms:.1f}ms (EMA: {self._infer_ms_ema:.1f}ms) target={new_frames}")

                for frame in frames:
                    await self.video_buffer.publish(frame)
                    # Use stats object for the watchdog to see
                    self.stats["last_video_send_epoch"] = time.time()
                    self.last_publish_epoch = time.time()
                    # Yield control to the event loop so recv() can process frames during a batch
                    await asyncio.sleep(0)

                if frames:
                    self.jobs += 1
                
                _publish_ms = (time.perf_counter() - _loop_start) * 1000
                # Update rolling average of loop latency
                self._infer_ms_ema = 0.8 * self._infer_ms_ema + 0.2 * _publish_ms
                print(f"[ENGINE] published {len(frames)} frames, total loop={_publish_ms:.1f}ms queue_now={self.video_buffer.queue.qsize()}")

            except Exception as e:
                self.last_error = repr(e)
                print(f"[engine] inference error: {e!r}")

            await asyncio.sleep(0.005)

    def status(self) -> dict:
        """Return engine diagnostics used by `/status`.

        Receives:
        - None.

        Returns:
        - Dict with counters and latest engine state/error.
        """

        return {
            "jobs": self.jobs,
            "last_publish_epoch": self.last_publish_epoch or None,
            "last_error": self.last_error or None,
            "avatar_frame_idx": int(self.avatar_time_secs * self.args.fps),
            "dropped_audio_ms_total": round(self.dropped_audio_ms_total, 1),
        }
