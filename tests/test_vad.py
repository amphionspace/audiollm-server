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

from backend.audio.vad import VADProcessor, trim_long_silence_for_asr  # noqa: E402
from backend.config import SAMPLE_RATE, load_config  # noqa: E402


class _ToggleBackend:
    """Deterministic VAD backend: returns whatever probability we set."""

    def __init__(self) -> None:
        self.prob = 0.0

    def process(self, frame: np.ndarray) -> float:
        return self.prob


class _AmplitudeBackend:
    """Deterministic VAD backend: speech iff the frame has real amplitude."""

    def process(self, frame: np.ndarray) -> float:
        return 1.0 if float(np.abs(frame).max()) > 0.1 else 0.0


def _make_processor(**kwargs: object) -> VADProcessor:
    # smoothing_alpha=0 -> smoothed == raw, so the toggle is exact.
    v = VADProcessor(smoothing_alpha=0.0, threshold=0.5, start_frames=2, **kwargs)
    v.vad = _ToggleBackend()
    v.smoothed_prob = None
    return v


def _tone(sec: float) -> np.ndarray:
    n = int(SAMPLE_RATE * sec)
    return (np.sin(np.linspace(0, 2 * np.pi * 220 * sec, n)) * 0.5).astype(np.float32)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * sec), dtype=np.float32)


def _patch_amplitude_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.audio.vad.VADProcessor._create_vad_backend",
        lambda self: _AmplitudeBackend(),
    )


def test_cut_fires_exactly_at_silence_frames() -> None:
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


def test_apply_config_pushes_per_connection_tunables() -> None:
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


def test_no_end_frames_attribute() -> None:
    v = VADProcessor()
    assert not hasattr(v, "end_frames")


def test_silence_removal_disabled_keeps_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_amplitude_vad(monkeypatch)
    pcm = np.concatenate([_tone(0.2), _silence(0.5), _tone(0.2)])
    cfg = load_config().override(asr_silence_removal_threshold_sec=0.0)

    res = trim_long_silence_for_asr(pcm, cfg)

    assert res.removed_ranges == 0
    assert res.removed_sec == 0.0
    np.testing.assert_array_equal(res.pcm, pcm)


def test_silence_removal_deletes_equal_threshold_internal_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_amplitude_vad(monkeypatch)
    first = _tone(0.2)
    pause = _silence(0.5)
    second = _tone(0.2)
    pcm = np.concatenate([first, pause, second])
    cfg = load_config().override(
        asr_silence_removal_threshold_sec=0.5,
        vad_threshold=0.5,
        vad_smoothing_alpha=0.0,
    )

    res = trim_long_silence_for_asr(pcm, cfg)

    assert res.removed_ranges == 1
    assert res.removed_sec == pytest.approx(0.5)
    np.testing.assert_array_equal(res.pcm, np.concatenate([first, second]))


def test_silence_removal_keeps_short_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_amplitude_vad(monkeypatch)
    pcm = np.concatenate([_tone(0.2), _silence(0.49), _tone(0.2)])
    cfg = load_config().override(
        asr_silence_removal_threshold_sec=0.5,
        vad_threshold=0.5,
        vad_smoothing_alpha=0.0,
    )

    res = trim_long_silence_for_asr(pcm, cfg)

    assert res.removed_ranges == 0
    np.testing.assert_array_equal(res.pcm, pcm)


def test_silence_removal_keeps_leading_trailing_and_all_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_amplitude_vad(monkeypatch)
    cfg = load_config().override(
        asr_silence_removal_threshold_sec=0.5,
        vad_threshold=0.5,
        vad_smoothing_alpha=0.0,
    )
    leading = np.concatenate([_silence(0.5), _tone(0.2)])
    trailing = np.concatenate([_tone(0.2), _silence(0.5)])
    all_silence = _silence(1.0)

    assert trim_long_silence_for_asr(leading, cfg).removed_ranges == 0
    assert trim_long_silence_for_asr(trailing, cfg).removed_ranges == 0
    res = trim_long_silence_for_asr(all_silence, cfg)

    assert res.removed_ranges == 0
    np.testing.assert_array_equal(res.pcm, all_silence)
