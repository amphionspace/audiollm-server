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
from .events import PartialSnapshot, SegmentReady, SpeechDropped, SpeechStarted

logger = logging.getLogger(__name__)

StreamEvent = SegmentReady | PartialSnapshot | SpeechStarted | SpeechDropped


@runtime_checkable
class AudioStream(Protocol):
    """Strategy interface for slicing the incoming PCM byte stream."""

    def configure(self, cfg: Config) -> None:
        """Apply (possibly per-session-overridden) Config knobs."""

    def feed(self, pcm_bytes: bytes) -> Iterable[StreamEvent]:
        """Push raw int16 little-endian PCM bytes; return zero or more events."""

    def flush(self, *, force: bool) -> Iterable[StreamEvent]:
        """Drain any remaining buffered audio (called on stop / disconnect).

        When ``force`` is True the implementation should always emit the
        residual audio (subject to non-empty); when False it may discard
        sub-threshold leftovers. Streams that announce ``SpeechStarted``
        must also emit ``SpeechDropped`` here whenever the in-flight
        utterance can't be recovered, so engines can retract any
        placeholder UI they painted on speech-start.
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

    def __init__(self, *, enable_partial: bool | None = None) -> None:
        self.vad = VADProcessor()
        self._pcm_carry: np.ndarray = np.empty(0, dtype=np.float32)
        # Running count of samples the VAD has processed across all feeds.
        # Used to stamp each segment with an approximate session-timeline
        # position (SegmentReady.start_ms/end_ms). Carry-over samples are
        # counted in the feed that actually processes them, so this stays a
        # continuous, monotonic clock.
        self._consumed_samples: int = 0
        self._cfg: Config | None = None
        self._first_partial_delay: float = 0.35
        self._partial_interval: float = 0.5
        self._last_partial_time: float = 0.0
        self._speech_started_time: float = 0.0
        self._sent_first_partial: bool = False
        # ``None`` means "follow cfg.enable_pseudo_stream"; callers that know
        # their downstream engine doesn't consume partials (e.g. emotion) can
        # pass ``False`` to skip the snapshot bookkeeping entirely regardless
        # of how the deployment toggles pseudo-stream globally.
        self._partial_override: bool | None = enable_partial
        self._enable_partial: bool = True
        # Tracks whether we've already emitted ``SpeechStarted`` for the
        # current in-flight utterance. Reset every time speech ends
        # (whether the resulting segment was usable or got dropped) so
        # the next silent->speaking transition fires exactly one event.
        self._announced_speech: bool = False
        self._ctc_tail_pause_logged: bool = False

    def configure(self, cfg: Config) -> None:
        self._cfg = cfg
        self.vad.apply_config(cfg)
        self._first_partial_delay = max(0, cfg.pseudo_stream_first_partial_ms) / 1000.0
        self._partial_interval = cfg.pseudo_stream_interval_ms / 1000.0
        if self._partial_override is None:
            # Partial snapshots only make sense if at least one ASR engine is
            # on AND pseudo-stream is enabled. Higher-level engines may further
            # suppress partials, but a stream that knows nothing about engines
            # uses the conservative default.
            self._enable_partial = bool(cfg.enable_pseudo_stream)
        else:
            self._enable_partial = self._partial_override

    @property
    def cfg(self) -> Config:
        if self._cfg is None:
            raise RuntimeError("VadSegmentedStream.configure() not called")
        return self._cfg

    def prefeed_partial_snapshot(self) -> PartialSnapshot | None:
        """Return the latest in-flight audio for low-latency local partials.

        Unlike normal partial snapshots, this is not throttled and is intended
        only for local streaming backends that separate audio feed from result
        publication.
        """
        if not self._enable_partial:
            return None
        now = time.monotonic()
        if self.vad.is_speaking:
            snapshot = self.vad.snapshot_incomplete_speech()
            if snapshot is None or snapshot.size == 0:
                return None
            return PartialSnapshot(
                pcm=snapshot,
                speech_started_at=self._speech_started_time or now,
                snapshot_at=now,
                is_first=not self._sent_first_partial,
            )
        return None

    def feed(self, pcm_bytes: bytes) -> list[StreamEvent]:
        events: list[StreamEvent] = []
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

        base = self._consumed_samples
        if hasattr(self.vad, "process_chunk"):
            vad_results = self.vad.process_chunk(pcm[:used])
        else:
            vad_results = []
            for i in range(0, used, hop):
                was_speaking = self.vad.is_speaking
                segment = self.vad.process(pcm[i : i + hop])
                now_speaking = self.vad.is_speaking
                vad_results.append((segment, was_speaking, now_speaking))

        for idx, (segment, was_speaking, now_speaking) in enumerate(vad_results):
            i = idx * hop

            # Silent -> speaking transition: announce immediately so any
            # downstream engine that wants to paint a placeholder ("识别
            # 中…") can do so before the user has even finished the
            # utterance.
            if not was_speaking and now_speaking and not self._announced_speech:
                self._announced_speech = True
                self._ctc_tail_pause_logged = False
                now = time.monotonic()
                self._speech_started_time = now
                self._last_partial_time = now
                self._sent_first_partial = False
                speech_timeline_ms = (base + i + hop) * 1000.0 / SAMPLE_RATE
                logger.info(
                    "VAD_TIMING event=speech_started timeline_ms=%.1f "
                    "consumed_ms=%.1f threshold=%.3f start_frames=%s "
                    "silence_frames=%s",
                    speech_timeline_ms,
                    base * 1000.0 / SAMPLE_RATE,
                    self.vad.threshold,
                    self.vad.start_frames,
                    self.vad.silence_frames,
                )
                events.append(SpeechStarted(started_at=now))

            if segment is None:
                continue

            # ``segment`` is the finalized speech buffer. Whether we
            # forward it as a usable ``SegmentReady`` or retract the
            # announcement via ``SpeechDropped``, the in-flight
            # announcement window has ended.
            announced = self._announced_speech
            self._announced_speech = False
            self._ctc_tail_pause_logged = False

            if len(segment) < min_samples:
                logger.info(
                    "Drop short segment (%.1fs)", len(segment) / SAMPLE_RATE
                )
                if announced:
                    events.append(SpeechDropped())
                continue
            end_sample = base + i + hop
            events.append(self._segment_with_timing(segment, end_sample))

        self._consumed_samples = base + used

        if self._enable_partial and self.vad.is_speaking:
            now = time.monotonic()
            # Gate on the cheap buffered length FIRST; only pay the O(n)
            # np.concatenate snapshot copy when we will actually emit a
            # partial. Computing the snapshot every feed made it O(n^2) over
            # an utterance (7% of BS52 feed-thread CPU in py-spy).
            buffered = self.vad.buffered_speech_samples()
            if buffered > 0:
                if not self._sent_first_partial:
                    first_samples = max(
                        self.vad.hop_size,
                        int(SAMPLE_RATE * self._first_partial_delay),
                    )
                    should_emit = buffered >= first_samples
                else:
                    should_emit = (
                        now - self._last_partial_time >= self._partial_interval
                        and buffered >= min_samples
                    )
                snapshot = self.vad.snapshot_incomplete_speech() if should_emit else None
                if should_emit and snapshot is not None:
                    is_first = not self._sent_first_partial
                    self._last_partial_time = now
                    self._sent_first_partial = True
                    if is_first:
                        logger.info(
                            "VAD_TIMING event=first_partial_snapshot "
                            "snapshot_audio_ms=%.1f speech_elapsed_ms=%.1f",
                            len(snapshot) * 1000.0 / SAMPLE_RATE,
                            (now - self._speech_started_time) * 1000.0,
                        )
                    events.append(
                        PartialSnapshot(
                            pcm=snapshot,
                            speech_started_at=self._speech_started_time,
                            snapshot_at=now,
                            is_first=is_first,
                        )
                    )

        return events

    def ctc_final_tail_pause_active(self) -> bool:
        cfg = self.cfg
        if not bool(getattr(cfg, "ctc_final_tail_pause_enabled", False)):
            return False
        if not self._enable_partial or not self._sent_first_partial:
            return False
        if not self.vad.is_speaking:
            return False
        threshold_ms = max(
            0, int(getattr(cfg, "ctc_final_tail_pause_silence_ms", 200) or 0)
        )
        threshold_frames = max(1, int(np.ceil(threshold_ms / self.vad.frame_ms)))
        return self.vad.silent_count >= threshold_frames

    def flush(self, *, force: bool) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        remaining = self.vad.flush()
        announced = self._announced_speech
        self._announced_speech = False
        if remaining is None or len(remaining) == 0:
            # No usable tail. If we'd announced speech-start the engine
            # is still showing a placeholder for it; let it retract.
            if announced:
                events.append(SpeechDropped())
            return events
        cfg = self.cfg
        min_samples = int(SAMPLE_RATE * cfg.min_segment_duration_ms / 1000)
        if not force and len(remaining) < min_samples:
            if announced:
                events.append(SpeechDropped())
            return events
        if force and len(remaining) < min_samples:
            # Force-flush still emits the residual but we drop the
            # placeholder announcement marker since this segment will
            # generate its own final.
            pass
        events.append(
            self._segment_with_timing(
                remaining, self._consumed_samples, is_stop_flush=force
            )
        )
        return events

    def _segment_with_timing(
        self,
        segment: np.ndarray,
        end_sample: int,
        *,
        is_stop_flush: bool = False,
    ) -> SegmentReady:
        start_sample = max(0, end_sample - len(segment))
        return SegmentReady(
            pcm=segment,
            is_stop_flush=is_stop_flush,
            start_ms=start_sample * 1000.0 / SAMPLE_RATE,
            end_ms=end_sample * 1000.0 / SAMPLE_RATE,
        )


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

    def feed(self, pcm_bytes: bytes) -> list[StreamEvent]:
        pcm = _pcm_bytes_to_float32(pcm_bytes)
        if pcm.size > 0:
            self._buffers.append(pcm)
        return []

    def flush(self, *, force: bool) -> list[StreamEvent]:
        if not self._buffers:
            return []
        merged = np.concatenate(self._buffers)
        self._buffers.clear()
        if merged.size == 0:
            return []
        return [SegmentReady(pcm=merged, is_stop_flush=force)]
