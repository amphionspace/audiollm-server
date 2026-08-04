from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.diarization.evaluation import (  # noqa: E402
    ReferenceTurn,
    ScoreRegion,
    fixed_mapping_score,
    hypothesis_annotation,
    project_single_role,
    reference_annotation,
    regions_timeline,
    standard_der,
)
from services.diarization.model import ModelTurn  # noqa: E402


def _score(
    reference: list[ReferenceTurn],
    hypothesis: list[ModelTurn],
    *,
    regions: list[ScoreRegion] | None = None,
) -> dict[str, float | bool]:
    score, _ = standard_der(
        reference_annotation(reference, uri="test"),
        hypothesis_annotation(hypothesis, uri="test"),
        uem=regions_timeline(regions or [ScoreRegion(0.0, 2.0)], uri="test"),
        collar_sec=0.0,
        skip_overlap=False,
    )
    return score


def test_standard_der_finds_permuted_speaker_mapping() -> None:
    reference = [
        ReferenceTurn(0.0, 1.0, "A"),
        ReferenceTurn(1.0, 2.0, "B"),
    ]
    hypothesis = [ModelTurn(0, 1000, 1), ModelTurn(1000, 2000, 0)]

    score = _score(reference, hypothesis)

    assert score["der"] == pytest.approx(0.0)
    assert score["reference_speaker_sec"] == pytest.approx(2.0)


def test_standard_der_reports_exact_missed_speech() -> None:
    score = _score(
        [ReferenceTurn(0.0, 2.0, "A")],
        [ModelTurn(0, 1000, 0)],
    )

    assert score["miss_sec"] == pytest.approx(1.0)
    assert score["der"] == pytest.approx(0.5)


def test_uem_excludes_errors_outside_scored_region() -> None:
    score = _score(
        [ReferenceTurn(0.0, 2.0, "A")],
        [ModelTurn(0, 1000, 0)],
        regions=[ScoreRegion(0.0, 1.0)],
    )

    assert score["der"] == pytest.approx(0.0)
    assert score["reference_speaker_sec"] == pytest.approx(1.0)


def test_fixed_mapping_bucket_does_not_hide_late_speaker_swap() -> None:
    reference = [
        ReferenceTurn(0.0, 1.0, "A"),
        ReferenceTurn(1.0, 2.0, "B"),
    ]
    hypothesis = [ModelTurn(0, 2000, 0)]

    score = fixed_mapping_score(
        reference,
        hypothesis,
        regions=[ScoreRegion(0.0, 2.0)],
        mapping={"speaker_0": "A"},
        start_sec=1.0,
        end_sec=2.0,
    )

    assert score["confusion_sec"] == pytest.approx(1.0)
    assert score["der"] == pytest.approx(1.0)


def test_single_role_projection_uses_dominant_speaker_for_overlap() -> None:
    turns = [ModelTurn(0, 1000, 0), ModelTurn(400, 600, 1)]

    assert project_single_role(turns, min_duration_ms=0) == [ModelTurn(0, 1000, 0)]


def test_single_role_projection_absorbs_short_flip_without_losing_time() -> None:
    turns = [
        ModelTurn(0, 400, 0),
        ModelTurn(400, 500, 1),
        ModelTurn(500, 1000, 0),
    ]

    assert project_single_role(turns, min_duration_ms=350) == [ModelTurn(0, 1000, 0)]
