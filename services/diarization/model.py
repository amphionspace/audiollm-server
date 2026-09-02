"""Low-latency Streaming Sortformer adapter with per-session speaker cache."""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MODEL_REPO = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"

SPEAKER_EMBEDDING_MODEL_REPO = "nvidia/speakerverification_en_titanet_large"
SPEAKER_EMBEDDING_MODEL_REVISION = "d6ba06bff20c64d51c946b676f4ec9b21fc45935"
SPEAKER_EMBEDDING_MODEL_FILENAME = "speakerverification_en_titanet_large.nemo"

SAMPLE_RATE = 16_000
MAX_SPEAKERS = 4
FRAME_MS = 80
CORE_FRAMES = 6
LEFT_CONTEXT_FRAMES = 1
RIGHT_CONTEXT_FRAMES = 7
SPEAKER_CACHE_FRAMES = 188
SPEAKER_CACHE_UPDATE_PERIOD_FRAMES = 144
SUBSAMPLING_FACTOR = 8
MEL_HOP_MS = 10
# The centered 512-point STFT needs 16 ms of real samples on each side.
# Two 10 ms hops keep the guard aligned with Mel frames before trimming.
FEATURE_GUARD_MS = 20
ACTIVITY_THRESHOLD = 0.5
MIN_FREE_GPU_GIB = 10.0
# A short micro-batch window groups same-phase streams without adding a full
# audio-frame (80 ms) of latency. The cap bounds one model invocation while
# still covering the current 16-session overload target in two batches.
DEFAULT_MAX_BATCH_SIZE = 8
DEFAULT_BATCH_WAIT_MS = 12.0
MAX_BATCH_SIZE_LIMIT = 32
MAX_BATCH_WAIT_MS = 100.0

_STREAMING_STATE_ATTRS = (
    "spkcache",
    "spkcache_lengths",
    "spkcache_preds",
    "fifo",
    "fifo_lengths",
    "fifo_preds",
    "spk_perm",
    "mean_sil_emb",
    "n_sil_frames",
)
_SCHEDULER_STOP = object()

logger = logging.getLogger(__name__)


class InsufficientGpuMemoryError(RuntimeError):
    """Permanent startup failure used to stop systemd restart loops."""


class SpeakerEmbeddingEngine:
    """Shared TitaNet speaker encoder for bounded, explicit RPC calls."""

    def __init__(self, model_path: str = "", *, device: str = "cuda") -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        if not model_path:
            model_path = hf_hub_download(
                repo_id=SPEAKER_EMBEDDING_MODEL_REPO,
                filename=SPEAKER_EMBEDDING_MODEL_FILENAME,
                revision=SPEAKER_EMBEDDING_MODEL_REVISION,
                local_files_only=os.getenv("HF_HUB_OFFLINE", "0") == "1",
            )
        logger.info("Loading speaker embedding checkpoint %s on %s", model_path, device)
        self.torch = torch
        self.device = torch.device(device)
        self.model = EncDecSpeakerLabelModel.restore_from(
            restore_path=model_path,
            map_location=self.device,
            strict=False,
        ).eval()
        self.model.to(self.device)
        featurizer = getattr(self.model.preprocessor, "featurizer", None)
        if featurizer is not None and hasattr(featurizer, "dither"):
            featurizer.dither = 0.0
        self._model_lock = threading.Lock()

    def extract(self, pcm_s16le: bytes) -> np.ndarray:
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise ValueError("speaker embedding requires aligned PCM_S16LE")
        waveform_np = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        if waveform_np.size == 0:
            raise ValueError("speaker embedding audio is empty")
        torch = self.torch
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)
        waveform_len = torch.tensor(
            [waveform_np.size],
            dtype=torch.long,
            device=self.device,
        )
        with self._model_lock, torch.inference_mode():
            _, embeddings = self.model(
                input_signal=waveform,
                input_signal_length=waveform_len,
            )
        if embeddings is None:
            raise RuntimeError("speaker embedding model returned no embedding")
        vector = embeddings.detach().float().cpu().numpy().reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("speaker embedding model returned an invalid vector")
        return np.asarray(vector / norm, dtype="<f4")


@dataclass(frozen=True)
class ModelTurn:
    start_ms: int
    end_ms: int
    speaker_index: int


@dataclass(frozen=True)
class ModelUpdate:
    finalized_through_ms: int
    turns: tuple[ModelTurn, ...]


@dataclass(frozen=True)
class _InferenceResult:
    state: object
    probabilities: np.ndarray


@dataclass
class _InferenceRequest:
    window: np.ndarray
    state: object
    real_core_samples: int
    trim_left: int
    trim_right: int
    left_offset: int
    right_offset: int
    future: Future[_InferenceResult] = field(default_factory=Future)

    def compatibility_key(self) -> tuple:
        """Return the dimensions that must match for synchronous NeMo state."""
        return (
            len(self.window),
            self.real_core_samples,
            self.trim_left,
            self.trim_right,
            self.left_offset,
            self.right_offset,
            _streaming_state_signature(self.state),
        )


class _BatchScheduler:
    """Collect compatible stream steps into bounded GPU micro-batches."""

    def __init__(
        self,
        run_batch: Callable[[list[_InferenceRequest]], list[_InferenceResult]],
        *,
        max_batch_size: int,
        batch_wait_ms: float,
    ) -> None:
        self._run_batch = run_batch
        self._max_batch_size = max_batch_size
        self._batch_wait_sec = batch_wait_ms / 1000.0
        self._queue: queue.Queue[object] = queue.Queue()
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="sortformer-batch-scheduler",
            daemon=True,
        )
        self._thread.start()

    def submit(self, request: _InferenceRequest) -> _InferenceResult:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Sortformer batch scheduler is closed")
            self._queue.put(request)
        return request.future.result()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_SCHEDULER_STOP)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is _SCHEDULER_STOP:
                return
            assert isinstance(first, _InferenceRequest)
            pending = [first]
            deadline = time.monotonic() + self._batch_wait_sec
            while len(pending) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is _SCHEDULER_STOP:
                    self._queue.put(_SCHEDULER_STOP)
                    break
                assert isinstance(item, _InferenceRequest)
                pending.append(item)

            groups: dict[tuple, list[_InferenceRequest]] = {}
            for request in pending:
                groups.setdefault(request.compatibility_key(), []).append(request)
            for requests in groups.values():
                try:
                    results = self._run_batch(requests)
                    if len(results) != len(requests):
                        raise RuntimeError("Sortformer batch returned wrong result count")
                except Exception as exc:
                    for request in requests:
                        request.future.set_exception(exc)
                else:
                    for request, result in zip(requests, results):
                        request.future.set_result(result)


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
        modules.fifo_len = SPEAKER_CACHE_FRAMES
        modules.spkcache_len = SPEAKER_CACHE_FRAMES
        modules.spkcache_update_period = SPEAKER_CACHE_UPDATE_PERIOD_FRAMES
        modules._check_streaming_parameters()
        featurizer = getattr(self.model.preprocessor, "featurizer", None)
        if featurizer is not None and hasattr(featurizer, "dither"):
            featurizer.dither = 0.0
        self._model_lock = threading.Lock()
        self.max_batch_size = _bounded_env_int(
            "DIARIZATION_MAX_BATCH_SIZE",
            default=DEFAULT_MAX_BATCH_SIZE,
            minimum=1,
            maximum=MAX_BATCH_SIZE_LIMIT,
        )
        self.batch_wait_ms = _bounded_env_float(
            "DIARIZATION_BATCH_WAIT_MS",
            default=DEFAULT_BATCH_WAIT_MS,
            minimum=0.0,
            maximum=MAX_BATCH_WAIT_MS,
        )
        self.batch_count = 0
        self.batch_item_count = 0
        self.max_batch_size_observed = 0
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
        self._batch_scheduler = _BatchScheduler(
            self._run_inference_batch,
            max_batch_size=self.max_batch_size,
            batch_wait_ms=self.batch_wait_ms,
        )
        logger.info(
            "Sortformer dynamic batching enabled; max_batch_size=%d batch_wait_ms=%.1f",
            self.max_batch_size,
            self.batch_wait_ms,
        )

    def new_stream(self) -> "SortformerStream":
        return SortformerStream(self)

    def infer(self, request: _InferenceRequest) -> _InferenceResult:
        return self._batch_scheduler.submit(request)

    def close(self) -> None:
        self._batch_scheduler.close()

    def _run_inference_batch(
        self, requests: list[_InferenceRequest]
    ) -> list[_InferenceResult]:
        if not requests:
            return []
        key = requests[0].compatibility_key()
        if any(request.compatibility_key() != key for request in requests[1:]):
            raise ValueError("incompatible Sortformer requests cannot share a batch")

        torch = self.torch
        started = time.monotonic()
        waveform = torch.from_numpy(
            np.stack([request.window for request in requests])
        ).to(self.device)
        waveform_len = torch.full(
            (len(requests),),
            len(requests[0].window),
            dtype=torch.long,
            device=self.device,
        )
        batched_state = _stack_streaming_states(
            [request.state for request in requests], torch=torch
        )
        first = requests[0]
        with self._model_lock, torch.inference_mode():
            features, feature_lengths = self.model.process_signal(
                audio_signal=waveform,
                audio_signal_length=waveform_len,
            )
            features = features[:, :, : feature_lengths.max()].transpose(1, 2)
            feature_end = (
                features.shape[1] - first.trim_right
                if first.trim_right
                else features.shape[1]
            )
            features = features[:, first.trim_left:feature_end, :]
            feature_lengths = (
                feature_lengths - first.trim_left - first.trim_right
            ).clamp(min=0)
            empty_preds = torch.zeros(
                (len(requests), 0, MAX_SPEAKERS), device=self.device
            )
            batched_state, chunk_preds = self.model.forward_streaming_step(
                processed_signal=features,
                processed_signal_length=feature_lengths,
                streaming_state=batched_state,
                total_preds=empty_preds,
                left_offset=first.left_offset,
                right_offset=first.right_offset,
            )

        valid_frames = max(
            1,
            math.ceil(
                first.real_core_samples / (SAMPLE_RATE * FRAME_MS / 1000)
            ),
        )
        probabilities = (
            chunk_preds[:, :valid_frames].detach().float().cpu().numpy()
        )
        states = _split_streaming_state(batched_state, count=len(requests))
        self.batch_count += 1
        self.batch_item_count += len(requests)
        if len(requests) > self.max_batch_size_observed:
            self.max_batch_size_observed = len(requests)
            logger.info(
                "Sortformer batch high-water mark: batch_size=%d infer_ms=%.1f",
                len(requests),
                (time.monotonic() - started) * 1000.0,
            )
        return [
            _InferenceResult(state=state, probabilities=probabilities[index])
            for index, state in enumerate(states)
        ]


class SortformerStream:
    """Incrementally preprocess PCM and advance NeMo's speaker-cache state."""

    def __init__(self, engine: SortformerEngine) -> None:
        self.engine = engine
        self._pcm = np.empty(0, dtype=np.float32)
        self._received_samples = 0
        self._next_core_sample = 0
        self._output_frames = 0
        self._finished = False
        self._state = engine.model.sortformer_modules.init_streaming_state(
            batch_size=1,
            async_streaming=engine.model.async_streaming,
            device=engine.device,
        )

    def feed(self, pcm_s16le: bytes) -> list[ModelUpdate]:
        if self._finished or not pcm_s16le:
            return []
        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        self._received_samples += len(samples)
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

        trim_left = round(
            (model_window_start - feature_window_start)
            / (SAMPLE_RATE * MEL_HOP_MS / 1000)
        )
        trim_right = round(
            (feature_window_end - model_window_end)
            / (SAMPLE_RATE * MEL_HOP_MS / 1000)
        )
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
        result = self.engine.infer(
            _InferenceRequest(
                window=window,
                state=self._state,
                real_core_samples=real_core_samples,
                trim_left=trim_left,
                trim_right=trim_right,
                left_offset=left_offset,
                right_offset=right_offset,
            )
        )
        self._state = result.state
        probs = result.probabilities
        frame_offset = self._output_frames
        self._output_frames += len(probs)
        finalized_through_ms = min(
            self._output_frames * FRAME_MS,
            self._received_samples * 1000 // SAMPLE_RATE,
        )
        turns = _probabilities_to_turns(probs, frame_offset=frame_offset)
        return ModelUpdate(
            finalized_through_ms=finalized_through_ms,
            turns=tuple(
                ModelTurn(
                    start_ms=turn.start_ms,
                    end_ms=min(turn.end_ms, finalized_through_ms),
                    speaker_index=turn.speaker_index,
                )
                for turn in turns
                if turn.start_ms < finalized_through_ms
            ),
        )


def _streaming_state_signature(state: object) -> tuple:
    signature = []
    for attr in _STREAMING_STATE_ATTRS:
        value = getattr(state, attr, None)
        if value is None:
            signature.append(None)
        else:
            signature.append(
                (tuple(value.shape[1:]), str(value.dtype), str(value.device))
            )
    return tuple(signature)


def _stack_streaming_states(states: list[object], *, torch) -> object:
    if not states:
        raise ValueError("cannot stack an empty streaming state list")
    batched = type(states[0])()
    for attr in _STREAMING_STATE_ATTRS:
        values = [getattr(state, attr, None) for state in states]
        if all(value is None for value in values):
            setattr(batched, attr, None)
            continue
        if any(value is None for value in values):
            raise ValueError(f"incompatible streaming state attribute: {attr}")
        setattr(batched, attr, torch.cat(values, dim=0))
    return batched


def _split_streaming_state(state: object, *, count: int) -> list[object]:
    states = [type(state)() for _ in range(count)]
    for attr in _STREAMING_STATE_ATTRS:
        value = getattr(state, attr, None)
        if value is None:
            for item in states:
                setattr(item, attr, None)
            continue
        if value.shape[0] != count:
            raise ValueError(f"invalid batched streaming state attribute: {attr}")
        for index, item in enumerate(states):
            # Clone so each session owns its state storage instead of retaining
            # the complete batch tensor through a view.
            setattr(item, attr, value[index : index + 1].clone())
    return states


def _bounded_env_int(
    name: str, *, default: int, minimum: int, maximum: int
) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if value < minimum or value > maximum:
        logger.warning(
            "%s=%d outside [%d, %d]; using %d",
            name,
            value,
            minimum,
            maximum,
            default,
        )
        return default
    return value


def _bounded_env_float(
    name: str, *, default: float, minimum: float, maximum: float
) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default
    if not math.isfinite(value) or value < minimum or value > maximum:
        logger.warning(
            "%s=%.3f outside [%.1f, %.1f]; using %.1f",
            name,
            value,
            minimum,
            maximum,
            default,
        )
        return default
    return value


def _probabilities_to_turns(
    probabilities: np.ndarray,
    *,
    frame_offset: int,
) -> list[ModelTurn]:
    turns: list[ModelTurn] = []
    if probabilities.ndim != 2:
        return turns
    for speaker in range(min(MAX_SPEAKERS, probabilities.shape[1])):
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
