"""Opt-in smoke test for the real NeMo checkpoint on GPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.diarization.model import SAMPLE_RATE, SortformerEngine  # noqa: E402


@pytest.mark.gpu
def test_real_sortformer_checkpoint_streams_to_final_watermark() -> None:
    model_path = os.getenv("DIARIZATION_GPU_TEST_MODEL", "").strip()
    if not model_path:
        pytest.skip("set DIARIZATION_GPU_TEST_MODEL to run the real GPU smoke test")
    assert Path(model_path).is_file(), model_path

    engine = SortformerEngine(model_path=model_path, device="cuda")
    stream = engine.new_stream()
    pcm = np.zeros(2 * SAMPLE_RATE, dtype="<i2").tobytes()
    updates = []
    for offset in range(0, len(pcm), 2560):
        updates.extend(stream.feed(pcm[offset : offset + 2560]))
    updates.extend(stream.finish())

    assert updates
    assert updates[-1].finalized_through_ms == 2000
    assert all(0 <= turn.speaker_index < 4 for update in updates for turn in update.turns)
