"""AudioStream strategies that decouple PCM ingestion from inference logic.

Two built-in strategies are provided:

- :class:`VadSegmentedStream` slices the input stream by VAD-detected speech
  segments and optionally emits :class:`PartialSnapshot` events while the user
  is still speaking (used for the ASR task).
- :class:`WholeUtteranceStream` accumulates everything and only emits a single
  :class:`SegmentReady` when the session is flushed (used for the emotion task).

Both classes implement the :class:`AudioStream` protocol so the
:class:`StreamingSession` can drive them without knowing the task semantics.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from ..audio.vad import VADProcessor
from ..config import SAMPLE_RATE, Config
from .events import PartialSnapshot, SegmentReady

logger = logging.getLogger(__name__)


@runtime_checkable
class AudioStream(Protocol):
    """Strategy interface for slicing the incoming PCM byte stream."""

    def configure(self, cfg: Config) -> None:
        """Apply (possibly per-session-overridden) Config knobs."""

    def feed(self, pcm_bytes: bytes) -> Iterable[SegmentReady | PartialSnapshot]:
        """Push raw int16 little-endian PCM bytes; return zero or more events."""

    def flush(self, *, force: bool) -> Iterable[SegmentReady]:
        """Drain any remaining buffered audio (called on stop / disconnect).

        When ``force`` is True the implementation should always emit the
        residual audio (subject to non-empty); when False it may discard
        sub-threshold leftovers.
        """


def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class VadSegmentedStream:
    """VAD-segmented stream with optional partial snapshots.

    Replicates the segmentation behavior of the original
    ``ASRStreamingSession``: feeds PCM frames into a :class:`VADProcessor`,
    emits a :class:`SegmentReady` when speech-to-silence is detected, and
    optionally emits a throttled :class:`PartialSnapshot` while the user is
    still speaking.
    """

    def __init__(self) -> None:
        self.vad = VADProcessor()
        self._pcm_carry: np.ndarray = np.empty(0, dtype=np.float32)
        self._cfg: Config | None = None
        self._partial_interval: float = 0.5
        self._last_partial_time: float = 0.0
        self._enable_partial: bool = True

    def configure(self, cfg: Config) -> None:
        self._cfg = cfg
        self._partial_interval = cfg.pseudo_stream_interval_ms / 1000.0
        # Partial snapshots only make sense if at least one ASR engine is on
        # AND pseudo-stream is enabled. Higher-level engines may further
        # suppress partials, but a stream that knows nothing about engines
        # uses the conservative default.
        self._enable_partial = bool(cfg.enable_pseudo_stream)

    @property
    def cfg(self) -> Config:
        if self._cfg is None:
            raise RuntimeError("VadSegmentedStream.configure() not called")
        return self._cfg

    def feed(self, pcm_bytes: bytes) -> list[SegmentReady | PartialSnapshot]:
        events: list[SegmentReady | PartialSnapshot] = []
        pcm = _pcm_bytes_to_float32(pcm_bytes)

        if self._pcm_carry.size > 0:
            pcm = np.concatenate([self._pcm_carry, pcm])

        hop = self.vad.hop_size
        used = (len(pcm) // hop) * hop
        self._pcm_carry = (
            pcm[used:].copy() if used < len(pcm) else np.empty(0, dtype=np.float32)
        )

        cfg = self.cfg
        min_samples = int(SAMPLE_RATE * cfg.min_segment_duration_ms / 1000)

        for i in range(0, used, hop):
            segment = self.vad.process(pcm[i : i + hop])
            if segment is None:
                continue
            if len(segment) < min_samples:
                logger.info(
                    "Drop short segment (%.1fs)", len(segment) / SAMPLE_RATE
                )
                continue
            events.append(SegmentReady(pcm=segment))

        if self._enable_partial and self.vad.is_speaking:
            now = time.monotonic()
            if now - self._last_partial_time >= self._partial_interval:
                snapshot = self.vad.snapshot_incomplete_speech()
                if snapshot is not None and len(snapshot) >= min_samples:
                    self._last_partial_time = now
                    events.append(PartialSnapshot(pcm=snapshot))

        return events

    def flush(self, *, force: bool) -> list[SegmentReady]:
        remaining = self.vad.flush()
        if remaining is None or len(remaining) == 0:
            return []
        cfg = self.cfg
        min_samples = int(SAMPLE_RATE * cfg.min_segment_duration_ms / 1000)
        if not force and len(remaining) < min_samples:
            return []
        return [SegmentReady(pcm=remaining, is_stop_flush=force)]


class WholeUtteranceStream:
    """Accumulates the entire audio stream and emits one segment on flush.

    Used by tasks like emotion recognition where the upstream client sends a
    complete utterance and the model takes the full clip as input.
    """

    def __init__(self) -> None:
        self._buffers: list[np.ndarray] = []
        self._cfg: Config | None = None

    def configure(self, cfg: Config) -> None:
        self._cfg = cfg

    def feed(self, pcm_bytes: bytes) -> list[SegmentReady | PartialSnapshot]:
        pcm = _pcm_bytes_to_float32(pcm_bytes)
        if pcm.size > 0:
            self._buffers.append(pcm)
        return []

    def flush(self, *, force: bool) -> list[SegmentReady]:
        if not self._buffers:
            return []
        merged = np.concatenate(self._buffers)
        self._buffers.clear()
        if merged.size == 0:
            return []
        return [SegmentReady(pcm=merged, is_stop_flush=force)]
