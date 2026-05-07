"""MuseTalk realtime inference engine wrapper."""

import argparse
import asyncio
import os
import shutil
import time

import cv2
import numpy as np
import torch

from .buffers import PcmRingBuffer, VideoFrameBuffer
from .models import AppArgs

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
        self.vad_hangover = 0
        self.vad_silence_streak = 0
        self.silero_vad_model = None
        self.silero_get_timestamps = None

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
        # Keep VAD on CPU to avoid contending with the main inference CUDA stream.
        self.silero_vad_model = self.silero_vad_model.to("cpu")
        self.silero_vad_model.eval()

    async def _topup_video_queue(self, target_frames: int) -> None:
        """Keep a small queue of duplicate frames ready during audio/input stalls."""

        target = max(1, min(int(target_frames), int(self.args.video_queue_size)))
        current = self.video_buffer.queue.qsize()
        if current >= target:
            return
        frame = self.video_buffer.last_frame
        for _ in range(target - current):
            await self.video_buffer.publish(frame)
            self.last_publish_epoch = time.time()

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
        rt = self.rt
        feature_ret = rt.audio_processor.get_audio_feature_from_array(
            pcm16k_window, sample_rate=16000, weight_dtype=rt.weight_dtype
        )
        if feature_ret is None:
            return []
        whisper_input_features, librosa_length = feature_ret
        whisper_chunks = rt.audio_processor.get_whisper_chunk(
            whisper_input_features,
            rt.device,
            rt.weight_dtype,
            rt.whisper,
            librosa_length,
            fps=self.args.fps,
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
        )
        if whisper_chunks is None or len(whisper_chunks) == 0:
            return []
        if new_frames > 0 and len(whisper_chunks) > new_frames:
            whisper_chunks = whisper_chunks[-new_frames:]
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

        window_samples = int((self.args.window_ms / 1000.0) * 16000)
        min_samples = int((self.args.min_window_ms / 1000.0) * 16000)
        max_advance_samples = int((self.args.max_advance_ms / 1000.0) * 16000)

        min_advance_samples = int((self.args.hop_ms / 1000.0) * 16000)

        while not self.stop_event.is_set():
            window, total = await self.pcm_ring.latest(window_samples)
            
            if self.last_total_samples < 0:
                new_samples = int(window.size)
            else:
                new_samples = max(0, int(total - self.last_total_samples))

            # If no new samples or not enough for a hop, wait for the event
            if new_samples < min_advance_samples:
                await self._topup_video_queue(target_frames=max(6, self.args.batch_size))
                await self.pcm_ring.wait_for_total_after(total, timeout=0.1)
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

                tensor_audio = torch.from_numpy(vad_audio_chunk)
                return self.silero_get_timestamps(
                    tensor_audio,
                    self.silero_vad_model,
                    sampling_rate=16000,
                    min_speech_duration_ms=20,
                    min_silence_duration_ms=10
                )

            speech_timestamps = await asyncio.to_thread(_run_vad, new_audio)
            
            speech_detected = len(speech_timestamps) > 0

            if speech_detected:
                self.vad_hangover = 15
                self.vad_silence_streak = 0
            else:
                self.vad_hangover = max(0, self.vad_hangover - 1)
                self.vad_silence_streak += 1

            if self.vad_hangover == 0:
                # For PersonaPlex-fed speech, treat VAD as advisory rather than a
                # hard gate. Silero false negatives can otherwise freeze the mouth
                # even while valid speech audio is playing cleanly downstream.
                hard_gate_audio = self.args.input_source != "mirror" or self.args.musetalk_only
                if hard_gate_audio:
                    window[-new_samples:] = 0.0

            new_frames = max(1, int(round((new_samples / 16000.0) * self.args.fps)))
            new_frames = min(new_frames, max(1, self.args.max_tail_frames))

            try:
                frames = await asyncio.to_thread(self._infer_window_frames, window, new_frames, new_samples)
                for frame in frames:
                    await self.video_buffer.publish(frame)
                    self.last_publish_epoch = time.time()
                if frames:
                    self.jobs += 1

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
