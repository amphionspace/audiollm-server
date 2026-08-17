import logging
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..config import HOP_SIZE, SAMPLE_RATE, Config, default_config

logger = logging.getLogger(__name__)


class _TenVadOnnx:
    """TEN VAD ONNX backend for Linux aarch64 builds.

    The PyPI ten-vad wheel ships x86_64 native binaries. The ONNX example
    builds a `ten_vad_python` extension that works on aarch64 while preserving
    the same frame-level VAD strategy in VADProcessor.
    """

    def __init__(self, hop_size: int, threshold: float):
        self.hop_size = hop_size
        self.threshold = threshold
        self._build_dir = self._resolve_build_dir()
        lib_dir = self._build_dir / "lib"
        if not lib_dir.is_dir():
            raise FileNotFoundError(f"TEN VAD ONNX lib dir not found: {lib_dir}")
        if not (self._build_dir / "onnx_model" / "ten-vad.onnx").is_file():
            raise FileNotFoundError(
                f"TEN VAD ONNX model not found under: {self._build_dir / 'onnx_model'}"
            )

        sys.path.insert(0, str(lib_dir))
        try:
            import ten_vad_python  # type: ignore[import-not-found]
        finally:
            try:
                sys.path.remove(str(lib_dir))
            except ValueError:
                pass

        cwd = Path.cwd()
        try:
            os.chdir(self._build_dir)
            self._vad = ten_vad_python.VAD(hop_size=hop_size, threshold=threshold)
        finally:
            os.chdir(cwd)

    @staticmethod
    def _resolve_build_dir() -> Path:
        candidates = []
        configured = os.getenv("TEN_VAD_ONNX_BUILD_DIR", "").strip()
        if configured:
            candidates.append(Path(configured))
        candidates.extend(
            [
                Path("/opt/ten-vad-onnx"),
                Path("/home/workspace/ten-vad/examples_onnx/python/build-linux"),
                Path(__file__).resolve().parents[2] / ".ten-vad-onnx" / "build-linux",
            ]
        )
        for path in candidates:
            if path.is_dir():
                return path.resolve()
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"TEN VAD ONNX build dir not found; searched: {searched}")

    def process(self, pcm_frame: np.ndarray) -> float:
        prob, _is_voice = self._vad.process(pcm_frame)
        return float(prob)

    def process_batch(self, pcm_chunk: np.ndarray) -> list[float]:
        """Process multiple hops in one GIL-released C++ call.

        Accepts float32 or int16. Float32 input is converted to int16
        inside the GIL-released C++ block for zero Python overhead.
        """
        if pcm_chunk.dtype == np.float32 and hasattr(self._vad, "process_batch_f32"):
            return [min(1.0, max(0.0, float(p))) for p in self._vad.process_batch_f32(pcm_chunk)]
        if pcm_chunk.dtype != np.int16:
            clipped = np.clip(pcm_chunk, -1.0, 1.0)
            pcm_chunk = (clipped * 32767.0).astype(np.int16)
        if hasattr(self._vad, "process_batch"):
            results = self._vad.process_batch(pcm_chunk)
            return [min(1.0, max(0.0, float(r[0]))) for r in results]
        hop = self.hop_size
        probs = []
        for i in range(len(pcm_chunk) // hop):
            prob, _ = self._vad.process(pcm_chunk[i * hop:(i + 1) * hop])
            probs.append(float(prob))
        return probs


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
        self._init_hop_size = hop_size
        self._init_threshold = threshold
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
        self.threshold = threshold
        self.smoothing_alpha = min(1.0, max(0.0, smoothing_alpha))
        self.start_frames = max(1, start_frames)
        self.silence_frames = max(1, math.ceil(silence_duration_ms / self.frame_ms))
        self.pre_speech_frames = max(1, math.ceil(pre_speech_ms / self.frame_ms))
        self.keep_tail_frames = max(0, math.ceil(keep_tail_ms / self.frame_ms))

    def apply_config(self, cfg: Config) -> None:
        """Apply per-connection VAD tunables without resetting audio buffers."""
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
        if isinstance(self.vad, _TenVadOnnx):
            # TEN VAD ONNX backend requires int16 PCM.
            if pcm_frame.dtype == np.int16:
                return pcm_frame
            clipped = np.clip(pcm_frame, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16, copy=False)
        # Generic fallback for non-batched TEN VAD variants.
        if pcm_frame.dtype == np.float32:
            return pcm_frame
        return pcm_frame.astype(np.float32, copy=False)

    def _create_vad_backend(self):
        try:
            return _TenVadOnnx(
                hop_size=self._init_hop_size,
                threshold=self._init_threshold,
            )
        except Exception as exc:
            raise RuntimeError(
                "TEN VAD ONNX backend is required for the delivery runtime; "
                "no RMS energy VAD fallback is available"
            ) from exc

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

    def process_chunk(self, pcm_chunk: np.ndarray, *, detection_only: bool = False):
        """Process multiple hops with batched ONNX + C++ state machine.

        Returns list of (segment_or_None, was_speaking, now_speaking) tuples.
        The entire ONNX inference + smoothing + speech detection runs in a
        single GIL-released C++ block for maximum thread parallelism.

        ``detection_only`` (design F12): callers that use this processor purely
        as a voice gate (e.g. ``AscendK2Stream``, which maintains its own speech
        buffer and ignores the returned segments) skip the per-hop Python audio
        buffer maintenance (``frame.copy()`` per 10 ms hop, ``audio_buffer``
        appends, per-segment ``np.concatenate``). Under BS52 that per-hop work
        runs on the single-process GIL for ~5200 hops/s and was a measurable
        share of the ~1.2%/s server real-time deficit that accumulates vad_lag.
        In this mode the segment element is always ``None`` and only the
        ``(was_speaking, now_speaking)`` transitions are meaningful.
        """
        hop = self.hop_size
        n_hops = len(pcm_chunk) // hop
        if n_hops == 0:
            return []

        chunk_aligned = pcm_chunk[:n_hops * hop]

        # Try C++ fast path: ONNX probabilities plus VAD state transitions run
        # in one GIL-released native call. Python only mirrors audio buffers so
        # finalized segments and partial snapshots keep the same semantics as
        # the frame-by-frame state machine.
        if (
            isinstance(self.vad, _TenVadOnnx)
            and hasattr(self.vad._vad, "process_chunk_f32")
        ):
            f32 = chunk_aligned if chunk_aligned.dtype == np.float32 else chunk_aligned.astype(np.float32)
            init_prob = self.smoothed_prob if self.smoothed_prob is not None else -1.0
            try:
                raw = self.vad._vad.process_chunk_f32(
                    f32, self.threshold, self.smoothing_alpha,
                    self.start_frames, self.silence_frames,
                    self.pre_speech_frames, self.keep_tail_frames,
                    self.is_speaking, init_prob,
                    self.silent_count, self.speech_count,
                )
            except TypeError:
                raw = None
            if raw is not None:
                # Last element is final state:
                # (smoothed, is_speaking, silent_count, speech_count)
                final_state = raw[-1]
                hop_results = raw[:-1]

                if detection_only:
                    # No Python-side buffer maintenance. hop_results tuples are
                    # (prob, was, now); the caller unpacks (_, was, now), so the
                    # prob slot doubles as the ignored segment slot. Update only
                    # the transition state the C++ machine advanced.
                    self.is_speaking = bool(final_state[1])
                    self.smoothed_prob = float(final_state[0])
                    self.silent_count = int(final_state[2])
                    self.speech_count = int(final_state[3])
                    return hop_results

                results = []
                for idx in range(n_hops):
                    prob, was, now = hop_results[idx]
                    frame = pcm_chunk[idx * hop:(idx + 1) * hop]
                    seg = self._update_state_from_cpp(frame, float(prob), bool(was), bool(now))
                    results.append((seg, bool(was), bool(now)))

                self.smoothed_prob = float(final_state[0])
                self.is_speaking = bool(final_state[1])
                self.silent_count = int(final_state[2])
                self.speech_count = int(final_state[3])
                return results

        # Fallback: Python state machine with batched ONNX probs
        if hasattr(self.vad, "process_batch"):
            raw_probs = self.vad.process_batch(chunk_aligned)
        else:
            vad_input = self._prepare_vad_input(chunk_aligned)
            raw_probs = [
                self._extract_prob(self.vad.process(vad_input[i*hop:(i+1)*hop]))
                for i in range(n_hops)
            ]

        results = []
        for idx in range(n_hops):
            was = self.is_speaking
            if detection_only:
                # Advance smoothing + speech-count state without buffer work.
                raw_prob = float(raw_probs[idx])
                if self.smoothed_prob is None:
                    self.smoothed_prob = raw_prob
                else:
                    a = self.smoothing_alpha
                    self.smoothed_prob = (a * self.smoothed_prob) + ((1.0 - a) * raw_prob)
                is_speech = self.smoothed_prob > self.threshold
                if not self.is_speaking:
                    self.speech_count = self.speech_count + 1 if is_speech else 0
                    if self.speech_count >= self.start_frames:
                        self.is_speaking = True
                        self.silent_count = 0
                else:
                    if is_speech:
                        self.silent_count = 0
                    else:
                        self.silent_count += 1
                        if self.silent_count >= self.silence_frames:
                            self.is_speaking = False
                            self.silent_count = 0
                            self.speech_count = 0
                results.append((None, was, self.is_speaking))
                continue
            frame = pcm_chunk[idx * hop:(idx + 1) * hop]
            seg = self._process_with_prob(frame, float(raw_probs[idx]))
            now = self.is_speaking
            results.append((seg, was, now))
        return results

    def _update_state_from_cpp(
        self,
        pcm_frame: np.ndarray,
        smoothed_prob: float,
        was_speaking: bool,
        now_speaking: bool,
    ) -> np.ndarray | None:
        """Mirror native VAD transitions into Python-owned audio buffers."""
        self.smoothed_prob = smoothed_prob
        frame_copy = pcm_frame.copy()

        if not was_speaking:
            self.pre_speech_buffer.append(frame_copy)
            if len(self.pre_speech_buffer) > self.pre_speech_frames:
                del self.pre_speech_buffer[0]

            if now_speaking:
                self.is_speaking = True
                self.audio_buffer.extend(self.pre_speech_buffer)
                self.pre_speech_buffer.clear()
            else:
                self.is_speaking = False
            return None

        self.audio_buffer.append(frame_copy)
        self.is_speaking = now_speaking
        if now_speaking:
            return None

        keep_tail = min(self.keep_tail_frames, self.silence_frames)
        trim = self.silence_frames - keep_tail
        if trim > 0:
            del self.audio_buffer[-trim:]
        segment = np.concatenate(self.audio_buffer)
        self.audio_buffer.clear()
        self.pre_speech_buffer.clear()
        return segment

    def _process_with_prob(self, pcm_frame: np.ndarray, raw_prob: float) -> np.ndarray | None:
        """State machine step using a pre-computed probability."""
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

        self.audio_buffer.append(frame_copy)
        if is_speech:
            self.silent_count = 0
        else:
            self.silent_count += 1
            if self.silent_count >= self.silence_frames:
                keep_tail = min(self.keep_tail_frames, self.silence_frames)
                trim = self.silence_frames - keep_tail
                if trim > 0:
                    del self.audio_buffer[-trim:]
                segment = np.concatenate(self.audio_buffer)
                self._reset()
                return segment
        return None

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

    def buffered_speech_samples(self) -> int:
        """Number of PCM samples buffered while speaking, without copying.

        Cheap O(nframes) length sum used to gate partial emission so the
        expensive ``np.concatenate`` in ``snapshot_incomplete_speech`` only
        runs when a partial is actually going to be emitted (avoids an
        O(n^2) per-feed copy of the whole growing speech buffer).
        """
        if not self.is_speaking or not self.audio_buffer:
            return 0
        return sum(len(f) for f in self.audio_buffer)

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
        probs[idx] = (
            vad.smoothed_prob if vad.smoothed_prob is not None else 0.0
        )

    frame_sec = hop / max(1, sample_rate)
    total_sec = n_frames * frame_sec
    voiced_frames = int((probs > prob_threshold).sum())
    voiced_sec = voiced_frames * frame_sec
    mean_prob = float(probs.mean())
    voiced_ratio = voiced_sec / total_sec if total_sec > 0 else 0.0
    return SpeechPresenceStats(total_sec, voiced_sec, mean_prob, voiced_ratio)
