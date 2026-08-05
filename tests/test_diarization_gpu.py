"""Opt-in smoke test for the real NeMo checkpoint on GPU."""

from __future__ import annotations

import os
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
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
    try:
        audio_path = os.getenv("DIARIZATION_GPU_TEST_AUDIO", "").strip()
        if audio_path:
            with wave.open(audio_path, "rb") as wav:
                assert wav.getframerate() == SAMPLE_RATE
                assert wav.getnchannels() == 1
                assert wav.getsampwidth() == 2
                pcm = wav.readframes(30 * SAMPLE_RATE)
        else:
            pcm = np.zeros(2 * SAMPLE_RATE, dtype="<i2").tobytes()
        chunks = [pcm[offset : offset + 2560] for offset in range(0, len(pcm), 2560)]

        reference_stream = engine.new_stream()
        reference_updates = []
        for chunk in chunks:
            reference_updates.extend(reference_stream.feed(chunk))
        reference_updates.extend(reference_stream.finish())

        streams = [engine.new_stream(), engine.new_stream()]
        batched_updates = [[], []]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for chunk in chunks:
                futures = [pool.submit(stream.feed, chunk) for stream in streams]
                for index, future in enumerate(futures):
                    batched_updates[index].extend(future.result())
            futures = [pool.submit(stream.finish) for stream in streams]
            for index, future in enumerate(futures):
                batched_updates[index].extend(future.result())

        assert reference_updates
        expected_duration_ms = len(pcm) * 1000 // (2 * SAMPLE_RATE)
        assert reference_updates[-1].finalized_through_ms == expected_duration_ms
        assert batched_updates == [reference_updates, reference_updates]
        assert engine.max_batch_size_observed >= 2
        assert all(
            0 <= turn.speaker_index < 4
            for update in reference_updates
            for turn in update.turns
        )
    finally:
        engine.close()
