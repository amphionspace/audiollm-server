"""Regression tests for diarization benchmark aggregation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "benchmark_diarization", ROOT / "scripts/benchmark_diarization.py"
)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)

STANDARD_SCORE_KEYS = BENCHMARK.STANDARD_SCORE_KEYS
_aggregate = BENCHMARK._aggregate


def _score(error: float, reference: float) -> dict[str, float]:
    return {
        "miss_sec": error,
        "false_alarm_sec": 0.0,
        "confusion_sec": 0.0,
        "reference_speaker_sec": reference,
        "der": error / reference,
    }


def _result(first_bucket: dict[str, float], second_bucket: dict[str, float]) -> dict[str, object]:
    zero = _score(0.0, 1.0)
    return {
        "duration_sec": 30.0,
        "performance": {"rtf": 0.1, "watermark_lag_p95_ms": 560.0},
        "scores": {
            "standard": {key: zero for key in STANDARD_SCORE_KEYS},
            "single_role_proxy": {key: zero for key in STANDARD_SCORE_KEYS},
            "fixed_mapping_buckets": {
                "standard": {
                    "origin_0_15": zero,
                    "origin_15_30": zero,
                },
                "single_role_proxy": {
                    "origin_0_15": first_bucket,
                    "origin_15_30": second_bucket,
                },
            },
        },
    }


def test_first_30_aggregation_combines_buckets_per_session() -> None:
    aggregate = _aggregate(
        [
            _result(_score(2.0, 1.0), _score(0.0, 9.0)),
            _result(_score(5.0, 10.0), _score(5.0, 10.0)),
        ]
    )

    first_30 = aggregate["single_role_proxy"]["first_30_fixed_mapping"]

    assert first_30["der"] == pytest.approx(0.4)
    assert first_30["macro_der"] == pytest.approx(0.35)
    assert first_30["worst_session_der"] == pytest.approx(0.5)
