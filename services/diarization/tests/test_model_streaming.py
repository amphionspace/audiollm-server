"""Unit tests for the low-latency Sortformer frontend."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.diarization.model import (  # noqa: E402
    CORE_FRAMES,
    FEATURE_GUARD_MS,
    FRAME_MS,
    MEL_HOP_MS,
    RIGHT_CONTEXT_FRAMES,
    SAMPLE_RATE,
    SortformerStream,
)


class _FakeModules:
    def init_streaming_state(self, **_kwargs):
        return object()


class _FakeModel:
    async_streaming = False

    def __init__(self) -> None:
        self.sortformer_modules = _FakeModules()
        self.audio_lengths: list[int] = []
        self.steps: list[tuple[int, int, int]] = []

    def process_signal(self, audio_signal, audio_signal_length):
        sample_count = int(audio_signal_length.item())
        self.audio_lengths.append(sample_count)
        frame_count = sample_count // (SAMPLE_RATE * MEL_HOP_MS // 1000)
        return torch.zeros((1, 128, frame_count)), torch.tensor([frame_count])

    def forward_streaming_step(
        self,
        *,
        processed_signal,
        processed_signal_length,
        streaming_state,
        total_preds,
        left_offset,
        right_offset,
    ):
        self.steps.append((int(processed_signal_length.item()), left_offset, right_offset))
        predictions = torch.zeros((1, CORE_FRAMES, 4))
        return streaming_state, torch.cat((total_preds, predictions), dim=1)


class _FakeEngine:
    def __init__(self) -> None:
        self.torch = torch
        self.device = torch.device("cpu")
        self.model = _FakeModel()
        self._model_lock = threading.Lock()


def _silence(duration_ms: int) -> bytes:
    return np.zeros(duration_ms * SAMPLE_RATE // 1000, dtype="<i2").tobytes()


def test_stream_waits_for_feature_guard_and_trims_guard_frames() -> None:
    engine = _FakeEngine()
    stream = SortformerStream(engine)  # type: ignore[arg-type]
    model_latency_ms = (CORE_FRAMES + RIGHT_CONTEXT_FRAMES) * FRAME_MS

    assert stream.feed(_silence(model_latency_ms)) == []
    assert len(stream.feed(_silence(FEATURE_GUARD_MS))) == 1
    assert engine.model.audio_lengths == [
        (model_latency_ms + FEATURE_GUARD_MS) * SAMPLE_RATE // 1000
    ]
    assert engine.model.steps == [(model_latency_ms // MEL_HOP_MS, 0, 56)]

    assert len(stream.feed(_silence(CORE_FRAMES * FRAME_MS))) == 1
    assert engine.model.audio_lengths[-1] == 1160 * SAMPLE_RATE // 1000
    assert engine.model.steps[-1] == (112, 8, 56)
