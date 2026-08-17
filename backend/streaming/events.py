"""Events produced by AudioStream strategies and consumed by TaskEngine."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class SegmentReady:
    """A finalized PCM segment ready for inference.

    ``pcm`` is float32, single channel, 16 kHz, range [-1, 1].
    ``is_stop_flush`` is True only when this event was produced by
    ``AudioStream.flush(force=True)`` after a ``stop`` message; engines may use
    this to distinguish the last segment of a session.

    ``start_ms`` / ``end_ms`` are the segment's approximate position within the
    session's audio timeline (measured from total PCM consumed by the stream,
    so they ignore VAD's end-of-speech detection lag and tail trimming). They
    default to ``None`` for streams/engines that don't track timing; only
    protocols that surface segment timing (e.g. AST v3 ``bg``/``ed``) read them.
    """

    pcm: np.ndarray
    is_stop_flush: bool = False
    start_ms: float | None = None
    end_ms: float | None = None
    queued_at: float = 0.0
    queue_depth_at_enqueue: int = 0
    dequeued_at: float = 0.0


@dataclass
class PartialSnapshot:
    """A snapshot of the audio currently buffered while user is speaking.

    Used by tasks that want to emit incremental ("pseudo-streaming") results
    before the speech segment is fully finalized. Streams that don't support
    pre-emption (e.g. WholeUtteranceStream) simply never produce this event.
    """

    pcm: np.ndarray
    speech_started_at: float | None = None
    snapshot_at: float = 0.0
    is_first: bool = False
    model_started_at: float = 0.0


@dataclass
class SpeechStarted:
    """VAD just transitioned silent -> speaking.

    Emitted exactly once per utterance, before any ``PartialSnapshot`` or
    ``SegmentReady``. Lets engines paint a placeholder UI (e.g. "识别中…")
    the moment the user opens their mouth instead of waiting for the
    segment to finalize. Engines that don't override the corresponding
    hook simply ignore the event.
    """

    started_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.monotonic()


@dataclass
class SpeechDropped:
    """The current utterance was abandoned without producing a segment.

    Emitted by the stream when speech ends but the resulting clip is
    rejected upstream (too short, sub-min duration, etc.) or when the
    session is force-flushed mid-speech with nothing useful to recover.
    Engines that announced a placeholder on ``SpeechStarted`` must
    retract it (e.g. send an empty ``final``) when they see this.
    """
