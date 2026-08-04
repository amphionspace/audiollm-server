#!/usr/bin/env python3
"""Benchmark Streaming Sortformer against RTTM/UEM with release-quality DER."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.diarization.evaluation import (  # noqa: E402
    ReferenceTurn,
    ScoreRegion,
    fixed_mapping_score,
    hypothesis_annotation,
    localize_reference,
    localize_regions,
    project_single_role,
    read_rttm,
    read_uem,
    reference_annotation,
    regions_timeline,
    standard_der,
)
from services.diarization.model import (  # noqa: E402
    MODEL_REPO,
    MODEL_REVISION,
    ModelTurn,
    SortformerEngine,
    _probabilities_to_turns,
)

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
STANDARD_SCORE_KEYS = (
    "no_collar_overlap_included",
    "collar_250ms_overlap_included",
    "collar_250ms_overlap_excluded",
)
BUCKETS = (
    ("origin_0_15", 0.0, 15.0),
    ("origin_15_30", 15.0, 30.0),
    ("steady_30_plus", 30.0, math.inf),
)


@dataclass(frozen=True)
class EvaluationItem:
    session_id: str
    audio_path: Path
    rttm_path: Path
    uem_path: Path | None = None
    audio_offset_sec: float = 0.0
    reference_offset_sec: float = 0.0
    duration_sec: float | None = None
    rttm_recording_id: str | None = None
    uem_recording_id: str | None = None
    expected_num_speakers: int | None = None


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_manifest(path: Path) -> list[EvaluationItem]:
    items: list[EvaluationItem] = []
    base = path.resolve().parent
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            audio_path = _resolve_path(row["audio_filepath"], base=base)
            rttm_path = _resolve_path(row["rttm_filepath"], base=base)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid manifest row: {exc}") from exc
        offset = float(row.get("offset", 0.0))
        items.append(
            EvaluationItem(
                session_id=str(row.get("uniq_id") or row.get("id") or audio_path.stem),
                audio_path=audio_path,
                rttm_path=rttm_path,
                uem_path=(
                    _resolve_path(row["uem_filepath"], base=base)
                    if row.get("uem_filepath")
                    else None
                ),
                audio_offset_sec=float(row.get("audio_offset", offset)),
                reference_offset_sec=float(row.get("reference_offset", offset)),
                duration_sec=(float(row["duration"]) if row.get("duration") is not None else None),
                rttm_recording_id=row.get("rttm_recording_id"),
                uem_recording_id=row.get("uem_recording_id"),
                expected_num_speakers=(
                    int(row["num_speakers"]) if row.get("num_speakers") is not None else None
                ),
            )
        )
    if not items:
        raise ValueError(f"{path}: manifest is empty")
    duplicate_ids = sorted(
        session_id
        for session_id in {item.session_id for item in items}
        if sum(item.session_id == session_id for item in items) > 1
    )
    if duplicate_ids:
        raise ValueError(f"{path}: duplicate session ids: {duplicate_ids}")
    return items


def _read_wav_window(item: EvaluationItem) -> tuple[bytes, float]:
    with wave.open(str(item.audio_path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            SAMPLE_RATE,
            1,
            SAMPLE_WIDTH,
        ):
            raise ValueError(f"{item.audio_path}: WAV must be 16 kHz mono signed 16-bit PCM")
        start_frame = round(item.audio_offset_sec * SAMPLE_RATE)
        if start_frame < 0 or start_frame > wav.getnframes():
            raise ValueError(f"{item.session_id}: audio offset is outside the WAV")
        available_frames = wav.getnframes() - start_frame
        requested_frames = (
            round(item.duration_sec * SAMPLE_RATE)
            if item.duration_sec is not None
            else available_frames
        )
        frame_count = min(available_frames, requested_frames)
        if frame_count <= 0:
            raise ValueError(f"{item.session_id}: selected audio window is empty")
        wav.setpos(start_frame)
        return wav.readframes(frame_count), frame_count / SAMPLE_RATE


def _coalesce(turns: list[ModelTurn]) -> list[ModelTurn]:
    merged: list[ModelTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start_ms, item.end_ms, item.speaker_index)):
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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _run_sidecar_stream(
    engine: SortformerEngine,
    pcm: bytes,
    *,
    chunk_ms: int,
) -> tuple[list[ModelTurn], dict[str, float | None]]:
    stream = engine.new_stream()
    chunk_bytes = chunk_ms * SAMPLE_RATE * SAMPLE_WIDTH // 1000
    turns: list[ModelTurn] = []
    step_latency_ms: list[float] = []
    watermark_lag_ms: list[float] = []
    received_ms = 0.0
    started = time.perf_counter()
    for offset in range(0, len(pcm), chunk_bytes):
        chunk = pcm[offset : offset + chunk_bytes]
        received_ms += len(chunk) / SAMPLE_WIDTH / SAMPLE_RATE * 1000
        step_started = time.perf_counter()
        updates = stream.feed(chunk)
        step_latency_ms.append((time.perf_counter() - step_started) * 1000)
        for update in updates:
            turns.extend(update.turns)
            watermark_lag_ms.append(received_ms - update.finalized_through_ms)
    finish_started = time.perf_counter()
    updates = stream.finish()
    finish_latency_ms = (time.perf_counter() - finish_started) * 1000
    for update in updates:
        turns.extend(update.turns)
        watermark_lag_ms.append(received_ms - update.finalized_through_ms)
    return _coalesce(turns), {
        "inference_sec": time.perf_counter() - started,
        "step_latency_p50_ms": statistics.median(step_latency_ms),
        "step_latency_p95_ms": _percentile(step_latency_ms, 0.95),
        "finish_latency_ms": finish_latency_ms,
        "watermark_lag_p95_ms": _percentile(watermark_lag_ms, 0.95),
    }


def _run_official_stream(
    engine: SortformerEngine,
    pcm: bytes,
) -> tuple[list[ModelTurn], dict[str, float | None]]:
    """Run NeMo's full-session forward_streaming as the implementation baseline."""

    torch = engine.torch
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0).to(engine.device)
    waveform_len = torch.tensor([len(samples)], device=engine.device)
    started = time.perf_counter()
    with engine._model_lock, torch.inference_mode():
        features, feature_lengths = engine.model.process_signal(
            audio_signal=waveform,
            audio_signal_length=waveform_len,
        )
        # NeMo's full ``forward_streaming`` owns its feature loader and expects
        # the preprocessor-native [batch, mel, time] layout. The sidecar's
        # lower-level ``forward_streaming_step`` expects [batch, time, mel].
        features = features[:, :, : feature_lengths.max()]
        probabilities = engine.model.forward_streaming(features, feature_lengths)
    inference_sec = time.perf_counter() - started
    turns = _probabilities_to_turns(probabilities[0].detach().float().cpu().numpy(), frame_offset=0)
    return _coalesce(turns), {
        "inference_sec": inference_sec,
        "step_latency_p50_ms": None,
        "step_latency_p95_ms": None,
        "finish_latency_ms": None,
        "watermark_lag_p95_ms": None,
    }


def _reference_speakers_in_regions(
    reference: list[ReferenceTurn], regions: list[ScoreRegion]
) -> list[str]:
    return sorted(
        {
            turn.speaker
            for turn in reference
            if any(
                turn.end_sec > region.start_sec and turn.start_sec < region.end_sec
                for region in regions
            )
        }
    )


def _score_variant(
    reference: list[ReferenceTurn],
    hypothesis: list[ModelTurn],
    *,
    regions: list[ScoreRegion],
    session_id: str,
) -> tuple[dict[str, dict[str, float | bool]], dict[Any, Any]]:
    reference_ann = reference_annotation(reference, uri=session_id)
    hypothesis_ann = hypothesis_annotation(hypothesis, uri=session_id)
    uem = regions_timeline(regions, uri=session_id)
    scores: dict[str, dict[str, float | bool]] = {}
    mapping: dict[Any, Any] = {}
    variants = (
        (STANDARD_SCORE_KEYS[0], 0.0, False),
        (STANDARD_SCORE_KEYS[1], 0.25, False),
        (STANDARD_SCORE_KEYS[2], 0.25, True),
    )
    for key, collar_sec, skip_overlap in variants:
        score, current_mapping = standard_der(
            reference_ann,
            hypothesis_ann,
            uem=uem,
            collar_sec=collar_sec,
            skip_overlap=skip_overlap,
        )
        scores[key] = score
        if key == STANDARD_SCORE_KEYS[0]:
            mapping = current_mapping
    return scores, mapping


def _bucket_scores(
    reference: list[ReferenceTurn],
    hypothesis: list[ModelTurn],
    *,
    regions: list[ScoreRegion],
    mapping: dict[Any, Any],
    duration_sec: float,
    exclude_overlap: bool,
) -> dict[str, dict[str, float | bool]]:
    scores: dict[str, dict[str, float | bool]] = {}
    for key, start_sec, nominal_end_sec in BUCKETS:
        end_sec = min(duration_sec, nominal_end_sec)
        if end_sec <= start_sec:
            continue
        scores[key] = fixed_mapping_score(
            reference,
            hypothesis,
            regions=regions,
            mapping=mapping,
            start_sec=start_sec,
            end_sec=end_sec,
            exclude_overlap=exclude_overlap,
        )
    return scores


def _evaluate_item(
    engine: SortformerEngine,
    item: EvaluationItem,
    *,
    chunk_ms: int,
    inference_mode: str,
    min_segment_duration_ms: int,
    allow_over_capacity: bool,
) -> dict[str, Any]:
    pcm, duration_sec = _read_wav_window(item)
    reference = localize_reference(
        read_rttm(item.rttm_path, recording_id=item.rttm_recording_id),
        offset_sec=item.reference_offset_sec,
        duration_sec=duration_sec,
    )
    regions = (
        localize_regions(
            read_uem(item.uem_path, recording_id=item.uem_recording_id),
            offset_sec=item.reference_offset_sec,
            duration_sec=duration_sec,
        )
        if item.uem_path
        else [ScoreRegion(0.0, duration_sec)]
    )
    if not regions:
        raise ValueError(f"{item.session_id}: UEM has no scored region in the audio window")
    reference_speakers = _reference_speakers_in_regions(reference, regions)
    if (
        item.expected_num_speakers is not None
        and len(reference_speakers) != item.expected_num_speakers
    ):
        raise ValueError(
            f"{item.session_id}: manifest num_speakers={item.expected_num_speakers}, "
            f"but RTTM/UEM contains {len(reference_speakers)}"
        )
    if len(reference_speakers) > 4 and not allow_over_capacity:
        raise ValueError(
            f"{item.session_id}: {len(reference_speakers)} scored speakers exceeds the "
            "4-speaker product boundary; use --allow-over-capacity only for diagnostics"
        )

    if inference_mode == "official":
        turns, performance = _run_official_stream(engine, pcm)
    else:
        turns, performance = _run_sidecar_stream(engine, pcm, chunk_ms=chunk_ms)
    projected = project_single_role(turns, min_duration_ms=min_segment_duration_ms)
    standard_scores, standard_mapping = _score_variant(
        reference,
        turns,
        regions=regions,
        session_id=item.session_id,
    )
    proxy_scores, proxy_mapping = _score_variant(
        reference,
        projected,
        regions=regions,
        session_id=f"{item.session_id}:single-role",
    )
    performance["rtf"] = float(performance["inference_sec"] or 0.0) / duration_sec
    return {
        "session_id": item.session_id,
        "audio_filepath": str(item.audio_path),
        "rttm_filepath": str(item.rttm_path),
        "uem_filepath": str(item.uem_path) if item.uem_path else None,
        "audio_offset_sec": item.audio_offset_sec,
        "reference_offset_sec": item.reference_offset_sec,
        "duration_sec": duration_sec,
        "reference_speakers": reference_speakers,
        "predicted_speakers": sorted({turn.speaker_index for turn in turns}),
        "performance": performance,
        "turns": [turn.__dict__ for turn in turns],
        "single_role_turns": [turn.__dict__ for turn in projected],
        "scores": {
            "standard": standard_scores,
            "single_role_proxy": proxy_scores,
            "fixed_mapping_buckets": {
                "standard": _bucket_scores(
                    reference,
                    turns,
                    regions=regions,
                    mapping=standard_mapping,
                    duration_sec=duration_sec,
                    exclude_overlap=False,
                ),
                "single_role_proxy": _bucket_scores(
                    reference,
                    projected,
                    regions=regions,
                    mapping=proxy_mapping,
                    duration_sec=duration_sec,
                    exclude_overlap=True,
                ),
            },
        },
    }


def _aggregate_components(scores: list[dict[str, Any]]) -> dict[str, float]:
    totals = {
        "miss_sec": sum(float(score["miss_sec"]) for score in scores),
        "false_alarm_sec": sum(float(score["false_alarm_sec"]) for score in scores),
        "confusion_sec": sum(float(score["confusion_sec"]) for score in scores),
        "reference_speaker_sec": sum(float(score["reference_speaker_sec"]) for score in scores),
    }
    error = totals["miss_sec"] + totals["false_alarm_sec"] + totals["confusion_sec"]
    totals["der"] = (
        error / totals["reference_speaker_sec"] if totals["reference_speaker_sec"] else 0.0
    )
    totals["macro_der"] = statistics.mean(float(score["der"]) for score in scores)
    totals["worst_session_der"] = max(float(score["der"]) for score in scores)
    return totals


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "session_count": len(results),
        "duration_sec": sum(float(result["duration_sec"]) for result in results),
        "standard": {},
        "single_role_proxy": {},
        "fixed_mapping_buckets": {"standard": {}, "single_role_proxy": {}},
        "performance": {
            "rtf_macro": statistics.mean(result["performance"]["rtf"] for result in results),
            "watermark_lag_p95_ms_max": max(
                (result["performance"]["watermark_lag_p95_ms"] or 0.0 for result in results),
                default=0.0,
            ),
        },
    }
    for family in ("standard", "single_role_proxy"):
        for key in STANDARD_SCORE_KEYS:
            aggregate[family][key] = _aggregate_components(
                [result["scores"][family][key] for result in results]
            )
        for bucket_key, _, _ in BUCKETS:
            bucket_scores = [
                result["scores"]["fixed_mapping_buckets"][family][bucket_key]
                for result in results
                if bucket_key in result["scores"]["fixed_mapping_buckets"][family]
            ]
            if bucket_scores:
                aggregate["fixed_mapping_buckets"][family][bucket_key] = _aggregate_components(
                    bucket_scores
                )
    first_30_scores = []
    for result in results:
        first_30_scores.extend(
            result["scores"]["fixed_mapping_buckets"]["single_role_proxy"][key]
            for key in ("origin_0_15", "origin_15_30")
            if key in result["scores"]["fixed_mapping_buckets"]["single_role_proxy"]
        )
    if first_30_scores:
        aggregate["single_role_proxy"]["first_30_fixed_mapping"] = _aggregate_components(
            first_30_scores
        )
    return aggregate


def _comparison(result: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_by_id = {item["session_id"]: item for item in result["sessions"]}
    baseline_by_id = {item["session_id"]: item for item in baseline["sessions"]}
    if current_by_id.keys() != baseline_by_id.keys():
        raise ValueError("comparison files must contain the same session ids")
    per_session = {}
    for session_id in sorted(current_by_id):
        current_der = current_by_id[session_id]["scores"]["standard"][STANDARD_SCORE_KEYS[0]]["der"]
        baseline_der = baseline_by_id[session_id]["scores"]["standard"][STANDARD_SCORE_KEYS[0]][
            "der"
        ]
        per_session[session_id] = {
            "current_der": current_der,
            "baseline_der": baseline_der,
            "absolute_percentage_point_difference": abs(current_der - baseline_der) * 100,
        }
    current_aggregate = result["aggregate"]["standard"][STANDARD_SCORE_KEYS[0]]["der"]
    baseline_aggregate = baseline["aggregate"]["standard"][STANDARD_SCORE_KEYS[0]]["der"]
    return {
        "baseline": str(baseline_path),
        "aggregate_absolute_percentage_point_difference": abs(
            current_aggregate - baseline_aggregate
        )
        * 100,
        "max_session_absolute_percentage_point_difference": max(
            item["absolute_percentage_point_difference"] for item in per_session.values()
        ),
        "sessions": per_session,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--wav", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rttm", type=Path)
    parser.add_argument("--uem", type=Path)
    parser.add_argument("--reference-offset-sec", type=float, default=0.0)
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--min-segment-duration-ms", type=int, default=350)
    parser.add_argument("--inference-mode", choices=("sidecar", "official"), default="sidecar")
    parser.add_argument("--allow-over-capacity", action="store_true")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.wav and not args.rttm:
        parser.error("--wav requires --rttm")
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    return args


def main() -> None:
    args = _parse_args()
    items = (
        _load_manifest(args.manifest)
        if args.manifest
        else [
            EvaluationItem(
                session_id=args.wav.stem,
                audio_path=args.wav,
                rttm_path=args.rttm,
                uem_path=args.uem,
                reference_offset_sec=args.reference_offset_sec,
            )
        ]
    )
    started = time.perf_counter()
    engine = SortformerEngine(str(args.model), device="cuda")
    model_load_sec = time.perf_counter() - started
    sessions = [
        _evaluate_item(
            engine,
            item,
            chunk_ms=args.chunk_ms,
            inference_mode=args.inference_mode,
            min_segment_duration_ms=args.min_segment_duration_ms,
            allow_over_capacity=args.allow_over_capacity,
        )
        for item in items
    ]
    result: dict[str, Any] = {
        "schema_version": 2,
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "inference_mode": args.inference_mode,
        "model_load_sec": model_load_sec,
        "sessions": sessions,
        "aggregate": _aggregate(sessions),
        "score_notes": {
            "standard": "continuous-time pyannote.metrics DER over RTTM/UEM",
            "fixed_mapping_buckets": (
                "exact continuous-time diagnostics using the full-session optimal mapping; "
                "bucket mappings are never re-optimized"
            ),
            "single_role_proxy": (
                "offline one-role projection without runtime VAD/k2 boundaries; not an exact "
                "end-to-end AST metric"
            ),
        },
    }
    torch = engine.torch
    result["gpu"] = {
        "peak_allocated_gib": torch.cuda.max_memory_allocated(engine.device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(engine.device) / 1024**3,
        "free_gib_after": torch.cuda.mem_get_info(engine.device)[0] / 1024**3,
    }
    if args.compare_to:
        result["comparison"] = _comparison(result, args.compare_to)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
