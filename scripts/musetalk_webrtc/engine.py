"""MuseTalk realtime inference engine wrapper."""

import argparse
import asyncio
import math
import shutil
import time

import cv2
import numpy as np
import torch
from einops import rearrange

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
        self.avatar_frame_idx = 0
        self.jobs = 0
        self.dropped_audio_ms_total = 0.0
        self.last_publish_epoch = 0.0
        self.last_error = ""
        self.prev_mouth_patch = None
        self._whisper_feature_cache = None
        self._cache_audio_samples = 0

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

        rt.vae, rt.unet, rt.pe = rt.load_all_model(
            unet_model_path=rt.args.unet_model_path,
            vae_type=rt.args.vae_type,
            unet_config=rt.args.unet_config,
            device=rt.device,
        )
        rt.timesteps = torch.tensor([0], device=rt.device)

        if rt.device.type == "cuda" and rt.args.use_fp16:
            rt.pe = rt.pe.half().to(rt.device)
            rt.vae.vae = rt.vae.vae.half().to(rt.device)
            rt.unet.model = rt.unet.model.half().to(rt.device)
            print("[engine] precision: fp16")
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
        if whisper_chunks is None or len(whisper_chunks) == 0:
            return []

        rt = self.rt
        combined_frames = []
        gen = rt.datagen(whisper_chunks, self.avatar.input_latent_list_cycle, self.args.batch_size)
        for whisper_batch, latent_batch in gen:
            audio_feature_batch = rt.pe(whisper_batch.to(rt.device))
            latent_batch = latent_batch.to(device=rt.device, dtype=rt.unet.model.dtype)
            pred_latents = rt.unet.model(
                latent_batch, rt.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents = pred_latents.to(device=rt.device, dtype=rt.vae.vae.dtype)
            recon = rt.vae.decode_latents(pred_latents)
            for res_frame in recon:
                base_i = self.avatar_frame_idx % len(self.avatar.frame_list_cycle)
                bbox = self.avatar.coord_list_cycle[base_i]
                ori_frame = self.avatar.frame_list_cycle[base_i].copy()
                x1, y1, x2, y2 = bbox
                try:
                    lip = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    self.avatar_frame_idx += 1
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
                self.avatar_frame_idx += 1

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
        hop_seconds = self.args.hop_ms / 1000.0
        while not self.stop_event.is_set():
            await asyncio.sleep(max(0.02, hop_seconds))
            window, total = await self.pcm_ring.latest(window_samples)
            if window.size < min_samples:
                continue
            if total == self.last_total_samples:
                continue

            if self.last_total_samples < 0:
                new_samples = int(window.size)
            else:
                new_samples = max(1, int(total - self.last_total_samples))
            self.last_total_samples = total

            if max_advance_samples > 0 and new_samples > max_advance_samples:
                dropped = (new_samples - max_advance_samples) * 1000.0 / 16000.0
                self.dropped_audio_ms_total += dropped
                new_samples = max_advance_samples

            # Fixes "same lips over and over" by publishing only newly advanced tail frames.
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
            "avatar_frame_idx": self.avatar_frame_idx,
            "dropped_audio_ms_total": round(self.dropped_audio_ms_total, 1),
        }
