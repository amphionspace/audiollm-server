"""Events produced by AudioStream strategies and consumed by TaskEngine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentReady:
    """A finalized PCM segment ready for inference.

    ``pcm`` is float32, single channel, 16 kHz, range [-1, 1].
    ``is_stop_flush`` is True only when this event was produced by
    ``AudioStream.flush(force=True)`` after a ``stop`` message; engines may use
    this to distinguish the last segment of a session.
    """

    pcm: np.ndarray
    is_stop_flush: bool = False


@dataclass
class PartialSnapshot:
    """A snapshot of the audio currently buffered while user is speaking.

    Used by tasks that want to emit incremental ("pseudo-streaming") results
    before the speech segment is fully finalized. Streams that don't support
    pre-emption (e.g. WholeUtteranceStream) simply never produce this event.
    """

    pcm: np.ndarray
