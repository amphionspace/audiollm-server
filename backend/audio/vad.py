import logging
import math

import numpy as np

try:
    from ten_vad import TenVad
except Exception:  # pragma: no cover - depends on optional native runtime
    TenVad = None

from ..config import (
    HOP_SIZE,
    SAMPLE_RATE,
    SILENCE_DURATION_MS,
    VAD_END_FRAMES,
    VAD_KEEP_TAIL_MS,
    VAD_PRE_SPEECH_MS,
    VAD_SMOOTHING_ALPHA,
    VAD_START_FRAMES,
    VAD_THRESHOLD,
)

logger = logging.getLogger(__name__)


class _EnergyVad:
    """Simple RMS-energy fallback VAD returning pseudo-probability [0, 1]."""

    def __init__(self, floor: float = 0.008, ceil: float = 0.06):
        self.floor = max(1e-6, floor)
        self.ceil = max(self.floor + 1e-6, ceil)

    def process(self, pcm_frame: np.ndarray) -> float:
        energy = float(np.sqrt(np.mean(np.square(pcm_frame), dtype=np.float32)))
        normalized = (energy - self.floor) / (self.ceil - self.floor)
        return float(min(1.0, max(0.0, normalized)))


def _patch_tenvad_destructor():
    """Guard against noisy AttributeError in ten-vad __del__."""
    if TenVad is None:
        return

    original_del = getattr(TenVad, "__del__", None)
    if not callable(original_del):
        return

    def _safe_del(self):
        # ten-vad may create partially initialized objects; skip unsafe cleanup.
        if not hasattr(self, "vad_library"):
            return
        try:
            original_del(self)
        except AttributeError:
            pass

    TenVad.__del__ = _safe_del


_patch_tenvad_destructor()


class VADProcessor:
    def __init__(
        self,
        hop_size: int = HOP_SIZE,
        threshold: float = VAD_THRESHOLD,
        silence_duration_ms: int = SILENCE_DURATION_MS,
        sample_rate: int = SAMPLE_RATE,
        smoothing_alpha: float = VAD_SMOOTHING_ALPHA,
        start_frames: int = VAD_START_FRAMES,
        pre_speech_ms: int = VAD_PRE_SPEECH_MS,
        end_frames: int = VAD_END_FRAMES,
        keep_tail_ms: int = VAD_KEEP_TAIL_MS,
    ):
        self.vad = self._create_vad_backend()
        backend_hop = getattr(self.vad, "hop_size", None)
        if isinstance(backend_hop, int) and backend_hop > 0:
            self.hop_size = backend_hop
        else:
            self.hop_size = hop_size
        self.sample_rate = max(1, sample_rate)
        self.frame_ms = (self.hop_size / self.sample_rate) * 1000.0
        self.threshold = threshold
        self.silence_frames = max(1, math.ceil(silence_duration_ms / self.frame_ms))
        self.end_frames = max(1, end_frames)
        self.pre_speech_frames = max(1, math.ceil(pre_speech_ms / self.frame_ms))
        self.keep_tail_frames = max(0, math.ceil(keep_tail_ms / self.frame_ms))
        self.start_frames = max(1, start_frames)
        self.smoothing_alpha = min(1.0, max(0.0, smoothing_alpha))
        self.audio_buffer: list[np.ndarray] = []
        self.pre_speech_buffer: list[np.ndarray] = []
        self.silent_count = 0
        self.speech_count = 0
        self.is_speaking = False
        self.smoothed_prob: float | None = None
        logger.info(
            "VAD backend=%s hop_size=%s frame_ms=%.1f pre_speech=%s silence=%s tail=%s",
            type(self.vad).__name__,
            self.hop_size,
            self.frame_ms,
            self.pre_speech_frames,
            self.silence_frames,
            self.keep_tail_frames,
        )

    def _prepare_vad_input(self, pcm_frame: np.ndarray) -> np.ndarray:
        """Adapt frame dtype for backend-specific requirements."""
        if TenVad is not None and isinstance(self.vad, TenVad):
            # ten-vad requires int16 PCM.
            if pcm_frame.dtype == np.int16:
                return pcm_frame
            clipped = np.clip(pcm_frame, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16, copy=False)
        # Energy fallback expects float-like input.
        if pcm_frame.dtype == np.float32:
            return pcm_frame
        return pcm_frame.astype(np.float32, copy=False)

    def _create_vad_backend(self):
        if TenVad is None:
            logger.warning(
                "ten-vad is unavailable; using fallback energy VAD. "
                "Install ten-vad and system libc++ (e.g. apt install libc++1)."
            )
            return _EnergyVad()

        try:
            return TenVad()
        except OSError as exc:
            logger.warning(
                "TEN VAD native library failed to load (%s). "
                "Using fallback energy VAD. "
                "Install system libc++ (e.g. apt install libc++1).",
                exc,
            )
            return _EnergyVad()

    def _extract_prob(self, value) -> float:
        """Normalize backend outputs to a single probability float in [0, 1]."""
        if isinstance(value, (tuple, list)):
            if not value:
                return 0.0
            # ten-vad may return tuples like (prob, state, ...)
            return self._extract_prob(value[0])
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            return self._extract_prob(float(value.reshape(-1)[0]))
        try:
            prob = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, prob))

    def process(self, pcm_frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame (hop_size samples, float32).
        Returns the full speech segment when speech-to-silence transition
        is detected, otherwise None.
        """
        vad_input = self._prepare_vad_input(pcm_frame)
        raw_prob = self._extract_prob(self.vad.process(vad_input))
        if self.smoothed_prob is None:
            self.smoothed_prob = raw_prob
        else:
            a = self.smoothing_alpha
            self.smoothed_prob = (a * self.smoothed_prob) + ((1.0 - a) * raw_prob)

        is_speech = self.smoothed_prob > self.threshold
        frame_copy = pcm_frame.copy()

        if not self.is_speaking:
            self.pre_speech_buffer.append(frame_copy)
            if len(self.pre_speech_buffer) > self.pre_speech_frames:
                del self.pre_speech_buffer[0]

            if is_speech:
                self.speech_count += 1
            else:
                self.speech_count = 0

            if self.speech_count >= self.start_frames:
                self.is_speaking = True
                self.silent_count = 0
                self.audio_buffer.extend(self.pre_speech_buffer)
                self.pre_speech_buffer.clear()
            return None

        # Speaking state.
        self.audio_buffer.append(frame_copy)
        if is_speech:
            self.silent_count = 0
        else:
            self.silent_count += 1
            end_threshold = max(self.silence_frames, self.end_frames)
            if self.silent_count >= end_threshold:
                # Trim trailing silence (keep a small tail for natural sound)
                keep_tail = min(self.keep_tail_frames, end_threshold)
                trim = end_threshold - keep_tail
                if trim > 0:
                    del self.audio_buffer[-trim:]
                segment = np.concatenate(self.audio_buffer)
                self._reset()
                return segment

        return None

    def snapshot_incomplete_speech(self) -> np.ndarray | None:
        """Return a copy of the PCM accumulated so far while speaking.

        Only meaningful when ``is_speaking`` is True and the buffer has
        accumulated at least *some* audio.  Returns ``None`` otherwise so
        the caller can skip pointless ASR requests.
        """
        if not self.is_speaking or not self.audio_buffer:
            return None
        return np.concatenate(self.audio_buffer)

    def flush(self) -> np.ndarray | None:
        """Flush any remaining buffered speech (e.g. on disconnect)."""
        if self.audio_buffer and self.is_speaking:
            segment = np.concatenate(self.audio_buffer)
            self._reset()
            return segment
        self._reset()
        return None

    def _reset(self):
        self.audio_buffer.clear()
        self.pre_speech_buffer.clear()
        self.silent_count = 0
        self.speech_count = 0
        self.is_speaking = False
        self.smoothed_prob = None
