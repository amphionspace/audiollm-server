"""Tests for async emotion HTTP job API."""

from __future__ import annotations

import asyncio
import io
import wave
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import SAMPLE_RATE
from backend.emotion.jobs import (
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_SUCCEEDED,
    EmotionJobStore,
    JobQueueFullError,
)
from backend.emotion.prompt import normalize_mode
from backend.emotion.service import build_final_emotion_payload, empty_final_emotion
from backend.main import app


def _make_wav_bytes(duration_sec: float = 0.5) -> bytes:
    n = int(SAMPLE_RATE * duration_sec)
    pcm = (np.sin(np.linspace(0, 800, n)) * 0.3 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.mark.asyncio
async def test_job_store_empty_audio_succeeds():
    store = EmotionJobStore()
    store.configure()
    store._queue_max = 4
    store._max_concurrent = 2

    job = await store.submit(b"", mode="ser", language="zh")
    for _ in range(50):
        await asyncio.sleep(0.05)
        polled = await store.get(job.job_id)
        if polled and polled.status == JOB_STATUS_SUCCEEDED:
            assert polled.result is not None
            assert polled.result["type"] == "final_emotion"
            assert polled.result["duration_sec"] == 0.0
            return
    pytest.fail("job did not complete")


@pytest.mark.asyncio
async def test_job_store_queue_full():
    store = EmotionJobStore()
    store.configure()
    store._queue_max = 1
    store._max_concurrent = 0
    store._semaphore = asyncio.Semaphore(0)

    await store.submit(_make_wav_bytes(0.1), mode="ser")
    with pytest.raises(JobQueueFullError):
        await store.submit(_make_wav_bytes(0.1), mode="ser")


def test_build_final_emotion_payload():
    payload = build_final_emotion_payload(
        {"label": "Happy", "text": "Happy"},
        mode="ser",
        duration_sec=1.5,
        language="zh",
    )
    assert payload["type"] == "final_emotion"
    assert payload["label"] == "Happy"
    assert payload["language"] == "zh"


def test_empty_final_emotion():
    payload = empty_final_emotion(mode="sec", language="en")
    assert payload["mode"] == "sec"
    assert payload["label"] == ""


def test_emotion_jobs_http_create_and_poll():
    client = TestClient(app)
    wav = _make_wav_bytes(0.3)

    with patch(
        "backend.emotion.service.query_emotion_model",
        new_callable=AsyncMock,
        return_value={"label": "Neutral", "text": "Neutral", "raw_text": "Neutral"},
    ):
        create = client.post(
            "/api/emotion/jobs",
            files={"audio": ("t.wav", wav, "audio/wav")},
            data={"mode": "ser", "language": "zh"},
        )
        assert create.status_code == 202
        body = create.json()
        assert body["status"] == JOB_STATUS_QUEUED
        job_id = body["job_id"]

        for _ in range(80):
            poll = client.get(f"/api/emotion/jobs/{job_id}")
            assert poll.status_code == 200
            data = poll.json()
            if data["status"] == JOB_STATUS_SUCCEEDED:
                assert data["result"]["label"] == "Neutral"
                return
            if data["status"] == JOB_STATUS_FAILED:
                pytest.fail(data.get("error"))
            import time

            time.sleep(0.05)
        pytest.fail("poll timed out")


def test_emotion_jobs_http_404():
    client = TestClient(app)
    resp = client.get("/api/emotion/jobs/em_missing123456")
    assert resp.status_code == 404


def test_emotion_protocol_only_accepts_ser_and_sec():
    assert normalize_mode("ser") == "ser"
    assert normalize_mode("sec") == "sec"
    for removed_mode in ("spec", "sepc", "para", "paralinguistic"):
        with pytest.raises(ValueError, match="mode must be ser or sec"):
            normalize_mode(removed_mode)


def test_emotion_spec_job_route_is_removed():
    client = TestClient(app)
    response = client.post(
        "/api/emotion-spec/jobs",
        files={"audio": ("t.wav", _make_wav_bytes(), "audio/wav")},
        data={"mode": "sec"},
    )
    assert response.status_code in {404, 405}
