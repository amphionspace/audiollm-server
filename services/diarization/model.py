"""Low-latency Streaming Sortformer adapter with per-session speaker cache."""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass

import numpy as np

MODEL_REPO = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"

SAMPLE_RATE = 16_000
FRAME_MS = 80
CORE_FRAMES = 6
LEFT_CONTEXT_FRAMES = 1
RIGHT_CONTEXT_FRAMES = 7
SUBSAMPLING_FACTOR = 8
MEL_HOP_MS = 10
# The centered 512-point STFT needs 16 ms of real samples on each side.
# Two 10 ms hops keep the guard aligned with Mel frames before trimming.
FEATURE_GUARD_MS = 20
ACTIVITY_THRESHOLD = 0.5
MIN_FREE_GPU_GIB = 10.0

logger = logging.getLogger(__name__)


class InsufficientGpuMemoryError(RuntimeError):
    """Permanent startup failure used to stop systemd restart loops."""


@dataclass(frozen=True)
class ModelTurn:
    start_ms: int
    end_ms: int
    speaker_index: int


@dataclass(frozen=True)
class ModelUpdate:
    finalized_through_ms: int
    turns: tuple[ModelTurn, ...]


class SortformerEngine:
    """Own the shared checkpoint; streaming tensor state belongs to streams."""

    def __init__(self, model_path: str = "", *, device: str = "cuda") -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from nemo.collections.asr.models import SortformerEncLabelModel

        if device.startswith("cuda"):
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes < MIN_FREE_GPU_GIB * 1024**3:
                raise InsufficientGpuMemoryError(
                    f"diarization requires {MIN_FREE_GPU_GIB:.0f} GiB free GPU memory; "
                    f"found {free_bytes / 1024**3:.1f} GiB"
                )
        if not model_path:
            model_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILENAME,
                revision=MODEL_REVISION,
                local_files_only=os.getenv("HF_HUB_OFFLINE", "0") == "1",
            )
        logger.info("Loading Sortformer checkpoint %s on %s", model_path, device)
        self.torch = torch
        self.device = torch.device(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        self.model = SortformerEncLabelModel.restore_from(
            restore_path=model_path,
            map_location=self.device,
            strict=False,
        ).eval()
        self.model.to(self.device)
        # The selected checkpoint is a streaming model, but make the runtime
        # contract explicit so feature normalization and inference stay in
        # the streaming path even if a future NeMo restore default changes.
        self.model.streaming_mode = True
        modules = self.model.sortformer_modules
        modules.chunk_len = CORE_FRAMES
        modules.chunk_left_context = LEFT_CONTEXT_FRAMES
        modules.chunk_right_context = RIGHT_CONTEXT_FRAMES
        modules.fifo_len = 188
        modules.spkcache_len = 188
        modules.spkcache_update_period = 144
        modules._check_streaming_parameters()
        featurizer = getattr(self.model.preprocessor, "featurizer", None)
        if featurizer is not None and hasattr(featurizer, "dither"):
            featurizer.dither = 0.0
        self._model_lock = threading.Lock()
        if device.startswith("cuda"):
            free_bytes, _ = torch.cuda.mem_get_info(device)
            logger.info(
                "Sortformer loaded; free_gpu_gib=%.1f reserved_gpu_gib=%.1f "
                "peak_allocated_gpu_gib=%.1f peak_reserved_gpu_gib=%.1f",
                free_bytes / 1024**3,
                torch.cuda.memory_reserved(self.device) / 1024**3,
                torch.cuda.max_memory_allocated(self.device) / 1024**3,
                torch.cuda.max_memory_reserved(self.device) / 1024**3,
            )
            if free_bytes < MIN_FREE_GPU_GIB * 1024**3:
                raise InsufficientGpuMemoryError(
                    "Sortformer loaded but less than 10 GiB GPU safety margin remains"
                )

    def new_stream(self) -> "SortformerStream":
        return SortformerStream(self)


class SortformerStream:
    """Incrementally preprocess PCM and advance NeMo's speaker-cache state."""

    def __init__(self, engine: SortformerEngine) -> None:
        self.engine = engine
        self._pcm = np.empty(0, dtype=np.float32)
        self._next_core_sample = 0
        self._output_frames = 0
        self._finished = False
        torch = engine.torch
        self._state = engine.model.sortformer_modules.init_streaming_state(
            batch_size=1,
            async_streaming=engine.model.async_streaming,
            device=engine.device,
        )
        self._empty_preds = torch.zeros((1, 0, 4), device=engine.device)

    def feed(self, pcm_s16le: bytes) -> list[ModelUpdate]:
        if self._finished or not pcm_s16le:
            return []
        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        self._pcm = np.concatenate((self._pcm, samples))
        return self._drain(force=False)

    def finish(self) -> list[ModelUpdate]:
        if self._finished:
            return []
        self._finished = True
        return self._drain(force=True)

    def _drain(self, *, force: bool) -> list[ModelUpdate]:
        core_samples = CORE_FRAMES * FRAME_MS * SAMPLE_RATE // 1000
        left_samples = LEFT_CONTEXT_FRAMES * FRAME_MS * SAMPLE_RATE // 1000
        right_samples = RIGHT_CONTEXT_FRAMES * FRAME_MS * SAMPLE_RATE // 1000
        feature_guard_samples = FEATURE_GUARD_MS * SAMPLE_RATE // 1000
        updates: list[ModelUpdate] = []
        while self._next_core_sample < len(self._pcm):
            available = len(self._pcm) - self._next_core_sample
            if not force and available < core_samples + right_samples + feature_guard_samples:
                break
            real_core_samples = min(core_samples, available)
            if real_core_samples <= 0:
                break
            updates.append(self._infer_step(real_core_samples=real_core_samples))
            self._next_core_sample += core_samples
        # Streaming state owns the long-term speaker history. Retain only the
        # raw left context needed by the next feature window so multi-hour
        # sessions stay memory-bounded.
        retain_from = max(
            0,
            self._next_core_sample - left_samples - feature_guard_samples,
        )
        if retain_from:
            self._pcm = self._pcm[retain_from:].copy()
            self._next_core_sample -= retain_from
        return updates

    def _infer_step(self, *, real_core_samples: int) -> ModelUpdate:
        torch = self.engine.torch
        core_samples = CORE_FRAMES * FRAME_MS * SAMPLE_RATE // 1000
        left_samples = LEFT_CONTEXT_FRAMES * FRAME_MS * SAMPLE_RATE // 1000
        right_samples = RIGHT_CONTEXT_FRAMES * FRAME_MS * SAMPLE_RATE // 1000
        feature_guard_samples = FEATURE_GUARD_MS * SAMPLE_RATE // 1000
        model_window_start = max(0, self._next_core_sample - left_samples)
        model_window_end = min(
            len(self._pcm),
            self._next_core_sample + core_samples + right_samples,
        )
        feature_window_start = max(0, model_window_start - feature_guard_samples)
        feature_window_end = min(
            len(self._pcm),
            model_window_end + feature_guard_samples,
        )
        window = self._pcm[feature_window_start:feature_window_end]
        required_core_end = self._next_core_sample - feature_window_start + core_samples
        if len(window) < required_core_end:
            window = np.pad(window, (0, required_core_end - len(window)))

        waveform = torch.from_numpy(window).unsqueeze(0).to(self.engine.device)
        waveform_len = torch.tensor([len(window)], device=self.engine.device)
        with self.engine._model_lock, torch.inference_mode():
            features, feature_lengths = self.engine.model.process_signal(
                audio_signal=waveform,
                audio_signal_length=waveform_len,
            )
            features = features[:, :, : feature_lengths.max()].transpose(1, 2)
            trim_left = round(
                (model_window_start - feature_window_start) / (SAMPLE_RATE * MEL_HOP_MS / 1000)
            )
            trim_right = round(
                (feature_window_end - model_window_end) / (SAMPLE_RATE * MEL_HOP_MS / 1000)
            )
            feature_end = features.shape[1] - trim_right if trim_right else features.shape[1]
            features = features[:, trim_left:feature_end, :]
            feature_lengths = (feature_lengths - trim_left - trim_right).clamp(min=0)
            left_offset = min(
                LEFT_CONTEXT_FRAMES * SUBSAMPLING_FACTOR,
                round(
                    (self._next_core_sample - model_window_start)
                    / (SAMPLE_RATE * MEL_HOP_MS / 1000)
                ),
            )
            future_samples = max(
                0,
                model_window_end - (self._next_core_sample + real_core_samples),
            )
            right_offset = min(
                RIGHT_CONTEXT_FRAMES * SUBSAMPLING_FACTOR,
                round(future_samples / (SAMPLE_RATE * MEL_HOP_MS / 1000)),
            )
            self._state, chunk_preds = self.engine.model.forward_streaming_step(
                processed_signal=features,
                processed_signal_length=feature_lengths,
                streaming_state=self._state,
                total_preds=self._empty_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )

        valid_frames = max(1, math.ceil(real_core_samples / (SAMPLE_RATE * FRAME_MS / 1000)))
        probs = chunk_preds[0, :valid_frames].detach().float().cpu().numpy()
        frame_offset = self._output_frames
        self._output_frames += len(probs)
        return ModelUpdate(
            finalized_through_ms=self._output_frames * FRAME_MS,
            turns=tuple(_probabilities_to_turns(probs, frame_offset=frame_offset)),
        )


def _probabilities_to_turns(
    probabilities: np.ndarray,
    *,
    frame_offset: int,
) -> list[ModelTurn]:
    turns: list[ModelTurn] = []
    if probabilities.ndim != 2:
        return turns
    for speaker in range(min(4, probabilities.shape[1])):
        active_start: int | None = None
        for index, probability in enumerate(probabilities[:, speaker]):
            active = float(probability) >= ACTIVITY_THRESHOLD
            if active and active_start is None:
                active_start = index
            if not active and active_start is not None:
                turns.append(
                    ModelTurn(
                        start_ms=(frame_offset + active_start) * FRAME_MS,
                        end_ms=(frame_offset + index) * FRAME_MS,
                        speaker_index=speaker,
                    )
                )
                active_start = None
        if active_start is not None:
            turns.append(
                ModelTurn(
                    start_ms=(frame_offset + active_start) * FRAME_MS,
                    end_ms=(frame_offset + len(probabilities)) * FRAME_MS,
                    speaker_index=speaker,
                )
            )
    return sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker_index))
