"""Continuous-time diarization scoring and AST single-role diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from .model import MAX_SPEAKERS, ModelTurn


@dataclass(frozen=True)
class ReferenceTurn:
    start_sec: float
    end_sec: float
    speaker: str


@dataclass(frozen=True)
class ScoreRegion:
    start_sec: float
    end_sec: float


def read_rttm(path: Path, *, recording_id: str | None = None) -> list[ReferenceTurn]:
    """Read one recording from an RTTM file using absolute source times."""

    records: list[tuple[str, ReferenceTurn]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 8 or fields[0] != "SPEAKER":
            raise ValueError(f"{path}:{line_number}: invalid RTTM SPEAKER row")
        start = float(fields[3])
        duration = float(fields[4])
        if duration <= 0:
            continue
        records.append(
            (
                fields[1],
                ReferenceTurn(start, start + duration, fields[7]),
            )
        )

    available = sorted({item[0] for item in records})
    selected = recording_id
    if selected is None:
        if len(available) > 1:
            raise ValueError(
                f"{path}: contains multiple recordings {available}; set rttm_recording_id"
            )
        selected = available[0] if available else None
    return [turn for current, turn in records if selected is None or current == selected]


def read_uem(path: Path, *, recording_id: str | None = None) -> list[ScoreRegion]:
    """Read one recording from a UEM file using absolute source times."""

    records: list[tuple[str, ScoreRegion]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number}: invalid UEM row")
        start = float(fields[2])
        end = float(fields[3])
        if end > start:
            records.append((fields[0], ScoreRegion(start, end)))

    available = sorted({item[0] for item in records})
    selected = recording_id
    if selected is None:
        if len(available) > 1:
            raise ValueError(
                f"{path}: contains multiple recordings {available}; set uem_recording_id"
            )
        selected = available[0] if available else None
    return [region for current, region in records if selected is None or current == selected]


def localize_reference(
    turns: Iterable[ReferenceTurn],
    *,
    offset_sec: float,
    duration_sec: float,
) -> list[ReferenceTurn]:
    """Clip absolute RTTM turns to an audio window and make them window-relative."""

    window_end = offset_sec + duration_sec
    localized = [
        ReferenceTurn(
            start_sec=max(offset_sec, turn.start_sec) - offset_sec,
            end_sec=min(window_end, turn.end_sec) - offset_sec,
            speaker=turn.speaker,
        )
        for turn in turns
        if turn.end_sec > offset_sec and turn.start_sec < window_end
    ]
    return sorted(localized, key=lambda turn: (turn.start_sec, turn.end_sec, turn.speaker))


def localize_regions(
    regions: Iterable[ScoreRegion],
    *,
    offset_sec: float,
    duration_sec: float,
) -> list[ScoreRegion]:
    """Clip absolute UEM regions to an audio window and merge overlaps."""

    window_end = offset_sec + duration_sec
    clipped = sorted(
        (
            ScoreRegion(
                max(offset_sec, region.start_sec) - offset_sec,
                min(window_end, region.end_sec) - offset_sec,
            )
            for region in regions
            if region.end_sec > offset_sec and region.start_sec < window_end
        ),
        key=lambda region: (region.start_sec, region.end_sec),
    )
    merged: list[ScoreRegion] = []
    for region in clipped:
        if merged and region.start_sec <= merged[-1].end_sec:
            previous = merged[-1]
            merged[-1] = ScoreRegion(previous.start_sec, max(previous.end_sec, region.end_sec))
        else:
            merged.append(region)
    return merged


def reference_annotation(turns: Iterable[ReferenceTurn], *, uri: str) -> Annotation:
    annotation = Annotation(uri=uri)
    for track, turn in enumerate(turns):
        annotation[Segment(turn.start_sec, turn.end_sec), track] = turn.speaker
    return annotation


def hypothesis_annotation(turns: Iterable[ModelTurn], *, uri: str) -> Annotation:
    annotation = Annotation(uri=uri)
    for track, turn in enumerate(turns):
        if turn.end_ms > turn.start_ms:
            annotation[Segment(turn.start_ms / 1000, turn.end_ms / 1000), track] = (
                f"speaker_{turn.speaker_index}"
            )
    return annotation


def regions_timeline(regions: Iterable[ScoreRegion], *, uri: str) -> Timeline:
    return Timeline(
        segments=[Segment(region.start_sec, region.end_sec) for region in regions],
        uri=uri,
    ).support()


def standard_der(
    reference: Annotation,
    hypothesis: Annotation,
    *,
    uem: Timeline,
    collar_sec: float,
    skip_overlap: bool,
) -> tuple[dict[str, float | bool], dict[Hashable, Hashable]]:
    """Score with pyannote.metrics and return JSON-safe components and mapping."""

    metric = DiarizationErrorRate(collar=collar_sec, skip_overlap=skip_overlap)
    details = metric(reference, hypothesis, uem=uem, detailed=True)
    result: dict[str, float | bool] = {
        "der": float(details["diarization error rate"]),
        "miss_sec": float(details["missed detection"]),
        "false_alarm_sec": float(details["false alarm"]),
        "confusion_sec": float(details["confusion"]),
        "reference_speaker_sec": float(details["total"]),
        "collar_sec": float(collar_sec),
        "exclude_overlap": bool(skip_overlap),
    }
    return result, metric.optimal_mapping(reference, hypothesis, uem=uem)


def project_single_role(
    turns: Iterable[ModelTurn],
    *,
    min_duration_ms: int,
) -> list[ModelTurn]:
    """Project overlapping model turns to the one-role-per-time AST constraint.

    This is a proxy because the offline evaluator does not know the runtime
    VAD/k2 segment boundaries. It preserves hypothesis silence and applies the
    same dominant-occupancy and short-flip rules within each connected speech
    component.
    """

    valid = sorted(
        (
            ModelTurn(int(turn.start_ms), int(turn.end_ms), int(turn.speaker_index))
            for turn in turns
            if (
                turn.end_ms > turn.start_ms
                and 0 <= int(turn.speaker_index) < MAX_SPEAKERS
            )
        ),
        key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker_index),
    )
    if not valid:
        return []

    components: list[list[ModelTurn]] = []
    component_end = -1
    for turn in valid:
        if components and turn.start_ms > component_end:
            components.append([turn])
        elif not components:
            components.append([turn])
        else:
            components[-1].append(turn)
        component_end = max(component_end, turn.end_ms)

    projected: list[ModelTurn] = []
    for component in components:
        occupancy: dict[int, int] = {}
        boundaries: set[int] = set()
        for turn in component:
            occupancy[turn.speaker_index] = occupancy.get(turn.speaker_index, 0) + (
                turn.end_ms - turn.start_ms
            )
            boundaries.update((turn.start_ms, turn.end_ms))
        intervals: list[ModelTurn] = []
        ordered = sorted(boundaries)
        for start_ms, end_ms in zip(ordered, ordered[1:]):
            active = [
                turn.speaker_index
                for turn in component
                if turn.end_ms > start_ms and turn.start_ms < end_ms
            ]
            if not active:
                continue
            speaker = min(set(active), key=lambda item: (-occupancy[item], item))
            candidate = ModelTurn(start_ms, end_ms, speaker)
            if (
                intervals
                and intervals[-1].speaker_index == speaker
                and intervals[-1].end_ms == start_ms
            ):
                previous = intervals[-1]
                intervals[-1] = ModelTurn(previous.start_ms, end_ms, speaker)
            else:
                intervals.append(candidate)
        projected.extend(_merge_short_turns(intervals, min_duration_ms=min_duration_ms))
    return projected


def _merge_short_turns(turns: list[ModelTurn], *, min_duration_ms: int) -> list[ModelTurn]:
    turns = list(turns)
    minimum = max(0, int(min_duration_ms))
    while len(turns) > 1:
        short_index = next(
            (index for index, turn in enumerate(turns) if turn.end_ms - turn.start_ms < minimum),
            None,
        )
        if short_index is None:
            break
        if short_index == 0:
            target = 1
        elif short_index == len(turns) - 1:
            target = short_index - 1
        else:
            left = turns[short_index - 1]
            right = turns[short_index + 1]
            target = (
                short_index - 1
                if left.end_ms - left.start_ms >= right.end_ms - right.start_ms
                else short_index + 1
            )
        low, high = sorted((short_index, target))
        replacement = ModelTurn(
            min(turns[short_index].start_ms, turns[target].start_ms),
            max(turns[short_index].end_ms, turns[target].end_ms),
            turns[target].speaker_index,
        )
        turns[low : high + 1] = [replacement]
        turns = _merge_adjacent(turns)
    return _merge_adjacent(turns)


def _merge_adjacent(turns: list[ModelTurn]) -> list[ModelTurn]:
    merged: list[ModelTurn] = []
    for turn in turns:
        if (
            merged
            and merged[-1].speaker_index == turn.speaker_index
            and merged[-1].end_ms == turn.start_ms
        ):
            previous = merged[-1]
            merged[-1] = ModelTurn(previous.start_ms, turn.end_ms, previous.speaker_index)
        else:
            merged.append(turn)
    return merged


def fixed_mapping_score(
    reference: Iterable[ReferenceTurn],
    hypothesis: Iterable[ModelTurn],
    *,
    regions: Iterable[ScoreRegion],
    mapping: dict[Hashable, Hashable],
    start_sec: float,
    end_sec: float,
    exclude_overlap: bool = False,
) -> dict[str, float | bool]:
    """Score an exact time bucket without re-optimizing the global mapping."""

    refs = list(reference)
    hyps = list(hypothesis)
    clipped_regions = [
        ScoreRegion(max(start_sec, region.start_sec), min(end_sec, region.end_sec))
        for region in regions
        if region.end_sec > start_sec and region.start_sec < end_sec
    ]
    boundaries = {start_sec, end_sec}
    for region in clipped_regions:
        boundaries.update((region.start_sec, region.end_sec))
    for turn in refs:
        if turn.end_sec > start_sec and turn.start_sec < end_sec:
            boundaries.update((max(start_sec, turn.start_sec), min(end_sec, turn.end_sec)))
    for turn in hyps:
        hyp_start = turn.start_ms / 1000
        hyp_end = turn.end_ms / 1000
        if hyp_end > start_sec and hyp_start < end_sec:
            boundaries.update((max(start_sec, hyp_start), min(end_sec, hyp_end)))

    miss = false_alarm = confusion = reference_time = 0.0
    ordered = sorted(boundaries)
    for interval_start, interval_end in zip(ordered, ordered[1:]):
        if interval_end <= interval_start:
            continue
        midpoint = (interval_start + interval_end) / 2
        if not any(region.start_sec <= midpoint < region.end_sec for region in clipped_regions):
            continue
        ref_set = {turn.speaker for turn in refs if turn.start_sec <= midpoint < turn.end_sec}
        if exclude_overlap and len(ref_set) > 1:
            continue
        hyp_set = {
            mapping.get(f"speaker_{turn.speaker_index}", f"__hyp_{turn.speaker_index}")
            for turn in hyps
            if turn.start_ms / 1000 <= midpoint < turn.end_ms / 1000
        }
        duration = interval_end - interval_start
        correct = len(ref_set & hyp_set)
        miss += max(0, len(ref_set) - len(hyp_set)) * duration
        false_alarm += max(0, len(hyp_set) - len(ref_set)) * duration
        confusion += (min(len(ref_set), len(hyp_set)) - correct) * duration
        reference_time += len(ref_set) * duration

    error = miss + false_alarm + confusion
    return {
        "der": error / reference_time if reference_time else 0.0,
        "miss_sec": miss,
        "false_alarm_sec": false_alarm,
        "confusion_sec": confusion,
        "reference_speaker_sec": reference_time,
        "exclude_overlap": exclude_overlap,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }
