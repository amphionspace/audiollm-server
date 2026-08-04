"""Speaker-turn timeline reconciliation and PCM splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import SAMPLE_RATE

if TYPE_CHECKING:
    from ..streaming.events import SegmentReady


@dataclass(frozen=True)
class SpeakerTurn:
    start_ms: int
    end_ms: int
    speaker_index: int


def _dominant_turn(
    turns: list[SpeakerTurn],
    start_ms: int,
    end_ms: int,
    occupancy_by_speaker: dict[int, int],
) -> SpeakerTurn | None:
    """Choose an active speaker, preferring greater whole-segment occupancy."""

    overlap_by_speaker: dict[int, int] = {}
    for turn in turns:
        overlap = max(0, min(end_ms, turn.end_ms) - max(start_ms, turn.start_ms))
        if overlap:
            overlap_by_speaker[turn.speaker_index] = (
                overlap_by_speaker.get(turn.speaker_index, 0) + overlap
            )
    if not overlap_by_speaker:
        return None
    speaker = min(
        overlap_by_speaker,
        key=lambda value: (
            -overlap_by_speaker[value],
            -occupancy_by_speaker.get(value, 0),
            value,
        ),
    )
    return SpeakerTurn(start_ms=start_ms, end_ms=end_ms, speaker_index=speaker)


def _merge_adjacent(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    merged: list[SpeakerTurn] = []
    for turn in turns:
        if merged and merged[-1].speaker_index == turn.speaker_index:
            previous = merged[-1]
            merged[-1] = SpeakerTurn(
                previous.start_ms,
                turn.end_ms,
                previous.speaker_index,
            )
        else:
            merged.append(turn)
    return merged


def _normalize_turns(
    turns: list[SpeakerTurn],
    *,
    segment_start_ms: int,
    segment_end_ms: int,
    min_duration_ms: int,
) -> list[SpeakerTurn]:
    """Return a gap-free, non-overlapping turn list inside one ASR segment.

    Sortformer can report overlapping speakers. AST v3 has one ``rl`` per
    candidate, so every boundary interval is assigned to the speaker with the
    greatest total overlap. Short flips are then absorbed into the longer
    adjacent turn; no PCM is discarded.
    """

    clipped = [
        SpeakerTurn(
            start_ms=max(segment_start_ms, int(turn.start_ms)),
            end_ms=min(segment_end_ms, int(turn.end_ms)),
            speaker_index=int(turn.speaker_index),
        )
        for turn in turns
        if 0 <= int(turn.speaker_index) < 4
        and int(turn.end_ms) > segment_start_ms
        and int(turn.start_ms) < segment_end_ms
    ]
    if not clipped:
        return []

    occupancy_by_speaker: dict[int, int] = {}
    for turn in clipped:
        occupancy_by_speaker[turn.speaker_index] = (
            occupancy_by_speaker.get(turn.speaker_index, 0)
            + turn.end_ms
            - turn.start_ms
        )

    boundaries = {segment_start_ms, segment_end_ms}
    for turn in clipped:
        boundaries.add(turn.start_ms)
        boundaries.add(turn.end_ms)
    ordered = sorted(boundaries)
    intervals: list[SpeakerTurn] = []
    for start_ms, end_ms in zip(ordered, ordered[1:]):
        if end_ms <= start_ms:
            continue
        chosen = _dominant_turn(
            clipped,
            start_ms,
            end_ms,
            occupancy_by_speaker,
        )
        if chosen is None:
            # Internal/edge gaps are assigned to the closest existing role.
            midpoint = (start_ms + end_ms) / 2
            nearest = min(
                clipped,
                key=lambda turn: (
                    min(abs(midpoint - turn.start_ms), abs(midpoint - turn.end_ms)),
                    turn.speaker_index,
                ),
            )
            chosen = SpeakerTurn(start_ms, end_ms, nearest.speaker_index)
        if intervals and intervals[-1].speaker_index == chosen.speaker_index:
            prev = intervals[-1]
            intervals[-1] = SpeakerTurn(prev.start_ms, chosen.end_ms, prev.speaker_index)
        else:
            intervals.append(chosen)

    minimum = max(0, int(min_duration_ms))
    while len(intervals) > 1:
        short_index = next(
            (
                index
                for index, turn in enumerate(intervals)
                if turn.end_ms - turn.start_ms < minimum
            ),
            None,
        )
        if short_index is None:
            break
        if short_index == 0:
            target = 1
        elif short_index == len(intervals) - 1:
            target = short_index - 1
        else:
            left = intervals[short_index - 1]
            right = intervals[short_index + 1]
            target = (
                short_index - 1
                if (left.end_ms - left.start_ms) >= (right.end_ms - right.start_ms)
                else short_index + 1
            )
        start = min(intervals[short_index].start_ms, intervals[target].start_ms)
        end = max(intervals[short_index].end_ms, intervals[target].end_ms)
        speaker = intervals[target].speaker_index
        low, high = sorted((short_index, target))
        intervals[low : high + 1] = [SpeakerTurn(start, end, speaker)]
        intervals = _merge_adjacent(intervals)

    return _merge_adjacent(intervals)


def split_segment_by_speaker(
    segment: SegmentReady,
    turns: list[SpeakerTurn],
    *,
    min_duration_ms: int,
) -> list[SegmentReady]:
    """Split a timed segment into gap-free speaker-attributed subsegments."""

    # Deferred to avoid ``streaming.__init__ -> session -> diarization.client``
    # while this module is still defining ``SpeakerTurn``.
    from ..streaming.events import SegmentReady

    if segment.start_ms is None or segment.end_ms is None or not turns:
        return [segment]
    start_ms = int(round(segment.start_ms))
    end_ms = int(round(segment.end_ms))
    normalized = _normalize_turns(
        turns,
        segment_start_ms=start_ms,
        segment_end_ms=end_ms,
        min_duration_ms=min_duration_ms,
    )
    if not normalized:
        return [segment]

    result: list[SegmentReady] = []
    for index, turn in enumerate(normalized):
        rel_start = max(0, int(round((turn.start_ms - start_ms) * SAMPLE_RATE / 1000)))
        rel_end = min(len(segment.pcm), int(round((turn.end_ms - start_ms) * SAMPLE_RATE / 1000)))
        if rel_end <= rel_start:
            continue
        result.append(
            SegmentReady(
                pcm=np.asarray(segment.pcm[rel_start:rel_end], dtype=np.float32),
                is_stop_flush=segment.is_stop_flush and index == len(normalized) - 1,
                id=f"{segment.id or 'segment'}:spk:{index}",
                start_ms=float(turn.start_ms),
                end_ms=float(turn.end_ms),
                speaker_index=turn.speaker_index,
            )
        )
    return result or [segment]
