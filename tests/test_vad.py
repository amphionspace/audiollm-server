"""Unit tests for VADProcessor segmentation knobs.

Focus
-----
1. The end-of-speech cut fires exactly at ``silence_frames`` silent frames.
   This locks in the removal of the old ``max(silence_frames, end_frames)``
   behavior (two params expressing the same physical quantity -> config
   spoofing). ``silence_duration_ms`` is now the single source of truth.
2. ``apply_config`` actually pushes per-connection tunables onto a live
   processor (the wiring that was missing and made VAD overrides no-op).
3. ``vad_end_frames`` is gone: the constructor rejects the kwarg and the
   instance exposes no ``end_frames`` attribute.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.audio.vad as vad_mod  # noqa: E402
from backend.audio.vad import VADProcessor  # noqa: E402
from backend.config import load_config  # noqa: E402


class _ToggleBackend:
    """Deterministic VAD backend: returns whatever probability we set."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.prob = 0.0
        self.hop_size = int(kwargs.get("hop_size", 256))

    def process(self, frame: np.ndarray) -> float:
        return self.prob


def _make_processor(**kwargs: object) -> VADProcessor:
    # smoothing_alpha=0 -> smoothed == raw, so the toggle is exact.
    v = VADProcessor(smoothing_alpha=0.0, threshold=0.5, start_frames=2, **kwargs)
    v.vad = _ToggleBackend()
    v.smoothed_prob = None
    return v


@pytest.fixture
def fake_ten_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad_mod, "_TenVadOnnx", _ToggleBackend)


def test_cut_fires_exactly_at_silence_frames(fake_ten_vad) -> None:
    v = _make_processor(silence_duration_ms=60, keep_tail_ms=0)
    frame = np.zeros(v.hop_size, dtype=np.float32)

    # Drive into the speaking state.
    v.vad.prob = 1.0
    for _ in range(v.start_frames):
        assert v.process(frame) is None
    assert v.is_speaking

    # Now feed silence: no cut until the silence_frames-th silent frame.
    v.vad.prob = 0.0
    for _ in range(v.silence_frames - 1):
        assert v.process(frame) is None
    seg = v.process(frame)
    assert seg is not None
    # Default end_frames used to be 18; if max() still governed, a 4-frame
    # silence window (60ms @16ms) would not have cut here.
    assert v.silence_frames < 18


def test_apply_config_pushes_per_connection_tunables(fake_ten_vad) -> None:
    v = VADProcessor()
    cfg = load_config().override(
        vad_threshold=0.21,
        silence_duration_ms=80,
        vad_smoothing_alpha=0.9,
        vad_start_frames=7,
        vad_pre_speech_ms=120,
        vad_keep_tail_ms=0,
    )
    v.apply_config(cfg)
    fm = v.frame_ms
    assert v.threshold == 0.21
    assert v.smoothing_alpha == 0.9
    assert v.start_frames == 7
    assert v.silence_frames == max(1, math.ceil(80 / fm))
    assert v.pre_speech_frames == max(1, math.ceil(120 / fm))
    assert v.keep_tail_frames == max(0, math.ceil(0 / fm))


def test_end_frames_param_is_rejected() -> None:
    with pytest.raises(TypeError):
        VADProcessor(end_frames=5)


def test_no_end_frames_attribute(fake_ten_vad) -> None:
    v = VADProcessor()
    assert not hasattr(v, "end_frames")


def test_ten_vad_initialization_failure_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenTenVad:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise FileNotFoundError("missing ten-vad runtime")

    monkeypatch.setattr(vad_mod, "_TenVadOnnx", _BrokenTenVad)
    with pytest.raises(RuntimeError, match="TEN VAD ONNX backend is required"):
        VADProcessor()
