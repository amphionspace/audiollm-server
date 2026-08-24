import logging
import math
from typing import NamedTuple

import numpy as np

try:
    from ten_vad import TenVad
except Exception:  # pragma: no cover - depends on optional native runtime
    TenVad = None

from ..config import HOP_SIZE, SAMPLE_RATE, Config, default_config

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
        threshold: float = default_config.vad_threshold,
        silence_duration_ms: int = default_config.silence_duration_ms,
        sample_rate: int = SAMPLE_RATE,
        smoothing_alpha: float = default_config.vad_smoothing_alpha,
        start_frames: int = default_config.vad_start_frames,
        pre_speech_ms: int = default_config.vad_pre_speech_ms,
        keep_tail_ms: int = default_config.vad_keep_tail_ms,
    ):
        self.vad = self._create_vad_backend()
        backend_hop = getattr(self.vad, "hop_size", None)
        if isinstance(backend_hop, int) and backend_hop > 0:
            self.hop_size = backend_hop
        else:
            self.hop_size = hop_size
        self.sample_rate = max(1, sample_rate)
        self.frame_ms = (self.hop_size / self.sample_rate) * 1000.0
        self._set_tunables(
            threshold=threshold,
            silence_duration_ms=silence_duration_ms,
            smoothing_alpha=smoothing_alpha,
            start_frames=start_frames,
            pre_speech_ms=pre_speech_ms,
            keep_tail_ms=keep_tail_ms,
        )
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

    def _set_tunables(
        self,
        *,
        threshold: float,
        silence_duration_ms: int,
        smoothing_alpha: float,
        start_frames: int,
        pre_speech_ms: int,
        keep_tail_ms: int,
    ) -> None:
        """Set the threshold-style segmentation knobs in one place.

        ``__init__`` and :meth:`apply_config` both route through here so the
        ms->frame conversion and clamping live once. ``frame_ms`` (derived
        from the backend hop at construction) must already be set. The VAD
        backend and any in-flight buffers/counters are deliberately left
        untouched, so this is safe to call mid-session.
        """
        self.threshold = threshold
        self.smoothing_alpha = min(1.0, max(0.0, smoothing_alpha))
        self.start_frames = max(1, start_frames)
        self.silence_frames = max(1, math.ceil(silence_duration_ms / self.frame_ms))
        self.pre_speech_frames = max(1, math.ceil(pre_speech_ms / self.frame_ms))
        self.keep_tail_frames = max(0, math.ceil(keep_tail_ms / self.frame_ms))

    def apply_config(self, cfg: Config) -> None:
        """Apply per-connection VAD tunables from a :class:`Config`.

        Called by the streaming layer's ``configure(cfg)`` after a client's
        ``start.config`` / ``parameter.asr_config`` override is merged, so
        these knobs actually take effect per connection instead of being
        frozen at construction to the process-wide defaults.
        """
        self._set_tunables(
            threshold=cfg.vad_threshold,
            silence_duration_ms=cfg.silence_duration_ms,
            smoothing_alpha=cfg.vad_smoothing_alpha,
            start_frames=cfg.vad_start_frames,
            pre_speech_ms=cfg.vad_pre_speech_ms,
            keep_tail_ms=cfg.vad_keep_tail_ms,
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
            if self.silent_count >= self.silence_frames:
                # Trim trailing silence (keep a small tail for natural sound)
                keep_tail = min(self.keep_tail_frames, self.silence_frames)
                trim = self.silence_frames - keep_tail
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


class SegmentVoiceEvidence(NamedTuple):
    accepted: bool
    reason: str
    speech_frames: int
    total_frames: int
    speech_ratio: float
    speech_ms: float
    max_prob: float
    mean_prob: float
    rms: float


class SilenceRemovalResult(NamedTuple):
    """PCM plus accounting for protected internal-silence removal."""

    pcm: np.ndarray
    removed_sec: float
    removed_ranges: int


class SpeechFilterResult(NamedTuple):
    """Admission evidence plus the speech-only PCM sent to AudioLLM."""

    pcm: np.ndarray
    evidence: SegmentVoiceEvidence
    input_samples: int
    kept_samples: int
    kept_ranges: int


class AsrAudioPreparationResult(NamedTuple):
    """Final-ASR preprocessing results from the shared boundary pipeline."""

    pcm: np.ndarray
    silence_removal: SilenceRemovalResult
    speech_filter: SpeechFilterResult


def filter_speech_for_asr(pcm: np.ndarray, cfg: Config) -> SpeechFilterResult:
    """Gate a segment and retain only VAD-supported speech regions.

    Local VAD and k2 remain endpoint authorities. This pass has a narrower
    responsibility at the AudioLLM boundary: reject clips with too little
    speech evidence, then remove unsupported non-speech regions from accepted
    clips. Each speech frame keeps the configured pre-speech/tail context so
    quiet onsets and word endings are not cut at the probability threshold.
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    input_samples = int(pcm.size)
    gate_enabled = cfg.asr_segment_voice_gate_enabled
    filter_enabled = cfg.asr_segment_voice_filter_enabled
    if not gate_enabled and not filter_enabled:
        evidence = SegmentVoiceEvidence(True, "disabled", 0, 0, 1.0, 0.0, 1.0, 1.0, 0.0)
        return SpeechFilterResult(
            pcm, evidence, input_samples, input_samples, int(input_samples > 0)
        )

    if pcm.size == 0:
        evidence = SegmentVoiceEvidence(
            not gate_enabled,
            "empty" if gate_enabled else "gate_disabled",
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        return SpeechFilterResult(pcm, evidence, input_samples, 0, 0)

    rms = float(np.sqrt(np.mean(np.square(pcm), dtype=np.float32)))
    if rms < cfg.asr_segment_voice_gate_min_rms:
        evidence = SegmentVoiceEvidence(
            not gate_enabled,
            "low_rms" if gate_enabled else "gate_disabled",
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            rms,
        )
        if not gate_enabled:
            return SpeechFilterResult(pcm, evidence, input_samples, input_samples, 1)
        return SpeechFilterResult(np.empty(0, dtype=np.float32), evidence, input_samples, 0, 0)

    processor = VADProcessor(
        threshold=(
            cfg.asr_segment_voice_gate_threshold
            if gate_enabled
            else cfg.asr_segment_voice_filter_threshold
        ),
        silence_duration_ms=cfg.silence_duration_ms,
        sample_rate=SAMPLE_RATE,
        smoothing_alpha=cfg.vad_smoothing_alpha,
        start_frames=1,
        pre_speech_ms=cfg.vad_pre_speech_ms,
        keep_tail_ms=cfg.vad_keep_tail_ms,
    )
    hop = processor.hop_size
    used = (pcm.size // hop) * hop
    if used <= 0:
        evidence = SegmentVoiceEvidence(
            not gate_enabled,
            "too_short" if gate_enabled else "gate_disabled",
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            rms,
        )
        if not gate_enabled:
            return SpeechFilterResult(pcm, evidence, input_samples, input_samples, 1)
        return SpeechFilterResult(np.empty(0, dtype=np.float32), evidence, input_samples, 0, 0)

    alpha = processor.smoothing_alpha
    smoothed: float | None = None
    probs: list[float] = []
    for i in range(0, used, hop):
        frame = pcm[i : i + hop]
        vad_input = processor._prepare_vad_input(frame)
        raw_prob = processor._extract_prob(processor.vad.process(vad_input))
        smoothed = raw_prob if smoothed is None else (alpha * smoothed) + ((1.0 - alpha) * raw_prob)
        probs.append(smoothed)

    total_frames = len(probs)
    gate_speech_flags = [prob > cfg.asr_segment_voice_gate_threshold for prob in probs]
    filter_speech_flags = [prob > cfg.asr_segment_voice_filter_threshold for prob in probs]
    evidence_flags = gate_speech_flags if gate_enabled else filter_speech_flags
    speech_frames = sum(evidence_flags)
    frame_ms = processor.frame_ms
    speech_ms = speech_frames * frame_ms
    speech_ratio = speech_frames / total_frames if total_frames else 0.0
    max_prob = max(probs) if probs else 0.0
    mean_prob = float(sum(probs) / total_frames) if total_frames else 0.0
    min_frames = max(1, math.ceil(cfg.asr_segment_voice_gate_min_ms / frame_ms))

    if not gate_enabled:
        reason = "gate_disabled"
        accepted = True
    elif speech_frames < min_frames:
        reason = "too_few_voice_frames"
        accepted = False
    elif speech_ratio < cfg.asr_segment_voice_gate_min_ratio:
        reason = "low_voice_ratio"
        accepted = False
    else:
        reason = "accepted"
        accepted = True

    evidence = SegmentVoiceEvidence(
        accepted,
        reason,
        speech_frames,
        total_frames,
        speech_ratio,
        speech_ms,
        max_prob,
        mean_prob,
        rms,
    )
    if not accepted:
        return SpeechFilterResult(np.empty(0, dtype=np.float32), evidence, input_samples, 0, 0)
    if not filter_enabled or not any(filter_speech_flags):
        return SpeechFilterResult(pcm, evidence, input_samples, input_samples, 1)

    pre_frames = max(0, math.ceil(cfg.asr_segment_voice_filter_pre_ms / frame_ms))
    tail_frames = max(0, math.ceil(cfg.asr_segment_voice_filter_tail_ms / frame_ms))
    keep_flags = [False] * total_frames
    for idx, is_speech in enumerate(filter_speech_flags):
        if not is_speech:
            continue
        start = max(0, idx - pre_frames)
        end = min(total_frames, idx + tail_frames + 1)
        keep_flags[start:end] = [True] * (end - start)

    keep_slices: list[np.ndarray] = []
    kept_ranges = 0
    idx = 0
    while idx < total_frames:
        if not keep_flags[idx]:
            idx += 1
            continue
        range_start = idx
        while idx < total_frames and keep_flags[idx]:
            idx += 1
        range_end = idx
        sample_start = range_start * hop
        sample_end = pcm.size if range_end == total_frames else range_end * hop
        keep_slices.append(pcm[sample_start:sample_end])
        kept_ranges += 1

    filtered = (
        np.concatenate(keep_slices).astype(np.float32, copy=False)
        if keep_slices
        else np.empty(0, dtype=np.float32)
    )
    return SpeechFilterResult(
        filtered,
        evidence,
        input_samples,
        int(filtered.size),
        kept_ranges,
    )


def trim_long_silence_for_asr(
    pcm: np.ndarray,
    cfg: Config,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> SilenceRemovalResult:
    """Delete internal long silence before final LLM ASR inference.

    ``cfg.asr_silence_removal_threshold_sec <= 0`` disables the pass. When
    enabled, only VAD-silent runs whose duration is greater than or equal to
    the configured threshold and that have speech on both sides are removed.
    Leading/trailing silence and all-silence clips are preserved so this pass
    cannot delete an entire utterance when VAD under-detects speech.
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    threshold_sec = float(getattr(cfg, "asr_silence_removal_threshold_sec", 0.0))
    if pcm.size == 0 or threshold_sec <= 0:
        return SilenceRemovalResult(pcm.astype(np.float32, copy=False), 0.0, 0)

    processor = VADProcessor(
        threshold=cfg.vad_threshold,
        silence_duration_ms=cfg.silence_duration_ms,
        sample_rate=sample_rate,
        smoothing_alpha=cfg.vad_smoothing_alpha,
        start_frames=1,
        pre_speech_ms=0,
        keep_tail_ms=0,
    )
    hop = processor.hop_size
    used = (pcm.size // hop) * hop
    if used <= 0:
        return SilenceRemovalResult(pcm.astype(np.float32, copy=False), 0.0, 0)

    min_silence_frames = max(
        1,
        math.ceil((threshold_sec * 1000.0) / processor.frame_ms),
    )
    frame_count = used // hop
    speech_flags: list[bool] = []
    smoothed: float | None = None
    for i in range(0, used, hop):
        frame = pcm[i : i + hop]
        vad_input = processor._prepare_vad_input(frame)
        raw_prob = processor._extract_prob(processor.vad.process(vad_input))
        if smoothed is None:
            smoothed = raw_prob
        else:
            alpha = processor.smoothing_alpha
            smoothed = (alpha * smoothed) + ((1.0 - alpha) * raw_prob)
        speech_flags.append(smoothed > processor.threshold)

    if not any(speech_flags):
        return SilenceRemovalResult(pcm.astype(np.float32, copy=False), 0.0, 0)

    has_speech_before: list[bool] = [False] * frame_count
    seen_speech = False
    for idx, is_speech in enumerate(speech_flags):
        has_speech_before[idx] = seen_speech
        seen_speech = seen_speech or is_speech

    has_speech_after: list[bool] = [False] * frame_count
    seen_speech = False
    for idx in range(frame_count - 1, -1, -1):
        has_speech_after[idx] = seen_speech
        seen_speech = seen_speech or speech_flags[idx]

    keep_slices: list[np.ndarray] = []
    keep_start = 0
    removed_samples = 0
    removed_ranges = 0
    idx = 0
    while idx < frame_count:
        if speech_flags[idx]:
            idx += 1
            continue
        run_start_frame = idx
        while idx < frame_count and not speech_flags[idx]:
            idx += 1
        run_end_frame = idx
        frames = run_end_frame - run_start_frame
        if frames < min_silence_frames:
            continue
        if not (has_speech_before[run_start_frame] and has_speech_after[run_end_frame - 1]):
            continue
        run_start = run_start_frame * hop
        run_end = run_end_frame * hop
        if keep_start < run_start:
            keep_slices.append(pcm[keep_start:run_start])
        removed_samples += run_end - run_start
        removed_ranges += 1
        keep_start = run_end

    if removed_ranges == 0:
        return SilenceRemovalResult(pcm.astype(np.float32, copy=False), 0.0, 0)
    if keep_start < pcm.size:
        keep_slices.append(pcm[keep_start:])

    out = np.concatenate(keep_slices).astype(np.float32, copy=False)
    return SilenceRemovalResult(
        out,
        removed_samples / max(1, sample_rate),
        removed_ranges,
    )


def prepare_audio_for_asr(
    pcm: np.ndarray,
    cfg: Config,
) -> AsrAudioPreparationResult:
    """Apply the shared final-ASR silence removal and speech policy."""

    silence_removal = trim_long_silence_for_asr(pcm, cfg)
    speech_filter = filter_speech_for_asr(silence_removal.pcm, cfg)
    return AsrAudioPreparationResult(
        speech_filter.pcm,
        silence_removal,
        speech_filter,
    )


def vad_trim_audio(
    pcm: np.ndarray,
    target_sec: float,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Keep up to ``target_sec`` of voiced audio from ``pcm`` via VAD.

    The input is walked hop-by-hop through a fresh :class:`VADProcessor` and
    each emitted speech segment is appended in order until the accumulated
    voiced duration reaches ``target_sec``. If the clip never transitioned
    to silence at the end (e.g. continuous speech up to the last sample),
    the processor's internal buffer is flushed so the tail isn't dropped.

    Rationale: callers typically only need a few seconds of clean speech
    from longer clips that have leading/trailing silence or a chatter
    preamble. Running VAD lets us throw away those boring segments before
    we hit the ``target_sec`` cap, rather than naively keeping the first N
    seconds (which may be silence) or the last N seconds (which may be
    mid-word).

    When VAD finds no voiced frames (e.g. extremely quiet microphone or a
    silent file) we fall back to the leading ``target_sec`` window so the
    caller still gets *something* to forward. The downstream duration guard
    will then reject the clip if it's too short after the cap.
    """
    if pcm.size == 0:
        return pcm.astype(np.float32, copy=False)
    target_samples = int(target_sec * sample_rate)
    if target_samples <= 0 or pcm.size <= target_samples:
        return pcm.astype(np.float32, copy=False)

    vad = VADProcessor(sample_rate=sample_rate)
    hop = vad.hop_size
    n_full = (pcm.size // hop) * hop

    collected: list[np.ndarray] = []
    accumulated = 0
    hit_target = False
    for i in range(0, n_full, hop):
        seg = vad.process(pcm[i : i + hop])
        if seg is not None:
            collected.append(seg)
            accumulated += seg.size
            if accumulated >= target_samples:
                hit_target = True
                break
    if not hit_target:
        tail = vad.flush()
        if tail is not None:
            collected.append(tail)
            accumulated += tail.size

    if not collected:
        return pcm[:target_samples].astype(np.float32, copy=False)

    out = np.concatenate(collected)
    if out.size > target_samples:
        out = out[:target_samples]
    return out.astype(np.float32, copy=False)


class SpeechPresenceStats(NamedTuple):
    """Per-segment speech-presence summary used as a second-stage gate.

    ``voiced_sec`` is the cumulative duration of frames whose smoothed VAD
    probability strictly exceeds the caller's threshold, ``total_sec`` is the
    analyzed duration (rounded down to whole hops), and ``mean_prob`` is the
    average smoothed probability across all analyzed frames. ``voiced_ratio``
    is ``voiced_sec / total_sec`` or ``0.0`` for empty input.
    """

    total_sec: float
    voiced_sec: float
    mean_prob: float
    voiced_ratio: float


def analyze_speech_presence(
    pcm: np.ndarray,
    *,
    prob_threshold: float = 0.6,
    sample_rate: int = SAMPLE_RATE,
) -> SpeechPresenceStats:
    """Compute speech-presence statistics for an already-segmented clip.

    Walks ``pcm`` hop-by-hop through a fresh :class:`VADProcessor` instance and
    records the post-smoothing probability at each step. The state-machine
    side effects (segment emission, internal buffers) are intentionally
    ignored — we only need the per-frame probabilities.

    Designed as a cheap (no extra network calls) second-stage gate on top of
    VAD-segmented audio: transient noise like keyboard taps tends to produce a
    short burst of high-prob frames inside a longer otherwise-silent segment,
    so its accumulated voiced duration stays well below that of even brief
    real speech. Callers typically pair this with a stricter ``prob_threshold``
    (e.g. 0.6) than the segmentation threshold (default 0.5).
    """
    if pcm.size == 0:
        return SpeechPresenceStats(0.0, 0.0, 0.0, 0.0)

    vad = VADProcessor()
    hop = vad.hop_size
    n_full = (pcm.size // hop) * hop
    if n_full <= 0:
        return SpeechPresenceStats(0.0, 0.0, 0.0, 0.0)

    n_frames = n_full // hop
    probs = np.empty(n_frames, dtype=np.float32)
    for idx, i in enumerate(range(0, n_full, hop)):
        vad.process(pcm[i : i + hop])
        probs[idx] = vad.smoothed_prob if vad.smoothed_prob is not None else 0.0

    frame_sec = hop / max(1, sample_rate)
    total_sec = n_frames * frame_sec
    voiced_frames = int((probs > prob_threshold).sum())
    voiced_sec = voiced_frames * frame_sec
    mean_prob = float(probs.mean())
    voiced_ratio = voiced_sec / total_sec if total_sec > 0 else 0.0
    return SpeechPresenceStats(total_sec, voiced_sec, mean_prob, voiced_ratio)
