from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.main as main_mod  # noqa: E402
from backend.audio.utils import pcm_to_wav_base64  # noqa: E402
from backend.config import SAMPLE_RATE, Config  # noqa: E402


def _wav_bytes(seconds: float = 1.2) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pcm = 0.2 * np.sin(2 * np.pi * 440 * t)
    return base64.b64decode(pcm_to_wav_base64(pcm.astype(np.float32)))


def _pcm_bytes(seconds: float = 1.2) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pcm = 0.2 * np.sin(2 * np.pi * 440 * t)
    return np.clip(pcm * 32767, -32768, 32767).astype(np.int16).tobytes()


def _mp3_bytes(seconds: float = 1.2) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not installed")
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=_wav_bytes(seconds),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg mp3 encode failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def test_enrollment_api_triton_store_does_not_use_local_store(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_upsert(pcm, **kwargs):
        captured["pcm_len"] = int(len(pcm))
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    def fail_local_store():
        raise AssertionError("local enrollment store should not be used")

    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda: Config(enable_triton_enrollment_store=True),
    )
    monkeypatch.setattr(main_mod, "upsert_triton_enrollment", fake_upsert)
    monkeypatch.setattr(main_mod, "get_enrollment_store", fail_local_store)

    with TestClient(main_mod.app) as client:
        resp = client.post(
            "/api/asr/enrollment",
            files={"audio": ("speaker.wav", _wav_bytes(), "audio/wav")},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["enrollment_id"]
    assert payload["duration_sec"] == 1.2
    kwargs = captured["kwargs"]
    assert kwargs["enrollment_id"] == payload["enrollment_id"]
    assert kwargs["enrollment_scope_id"] == "default"
    assert kwargs["sample_rate"] == SAMPLE_RATE
    assert captured["pcm_len"] == int(SAMPLE_RATE * 1.2)


def test_enrollment_status_local_found_and_not_found(monkeypatch):
    monkeypatch.setattr(main_mod, "load_config", lambda: Config())

    with TestClient(main_mod.app) as client:
        create = client.post(
            "/api/asr/enrollment",
            files={"audio": ("speaker.wav", _wav_bytes(), "audio/wav")},
        )
        enrollment_id = create.json()["enrollment_id"]

        found = client.get(f"/api/asr/enrollment/{enrollment_id}")
        missing = client.get("/api/asr/enrollment/missing-speaker")

    assert found.status_code == 200
    assert found.json() == {
        "enrollment_id": enrollment_id,
        "available": True,
        "reason": "ok",
    }
    assert missing.status_code == 200
    assert missing.json() == {
        "enrollment_id": "missing-speaker",
        "available": False,
        "reason": "not_found",
    }


def test_enrollment_status_triton_store(monkeypatch):
    async def fake_get(**kwargs):
        assert kwargs["enrollment_id"] == "speaker-1"
        assert kwargs["enrollment_scope_id"] == "default"
        return {"status": "ok", "available": True, "reason": "ok"}

    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda: Config(enable_triton_enrollment_store=True),
    )
    monkeypatch.setattr(main_mod, "get_triton_enrollment", fake_get)

    with TestClient(main_mod.app) as client:
        resp = client.get("/api/asr/enrollment/speaker-1")

    assert resp.status_code == 200
    assert resp.json() == {
        "enrollment_id": "speaker-1",
        "available": True,
        "reason": "ok",
    }


def test_enrollment_status_triton_upstream_unavailable(monkeypatch):
    async def fake_get(**_kwargs):
        raise RuntimeError("management down")

    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda: Config(enable_triton_enrollment_store=True),
    )
    monkeypatch.setattr(main_mod, "get_triton_enrollment", fake_get)

    with TestClient(main_mod.app) as client:
        resp = client.get("/api/asr/enrollment/speaker-1")

    assert resp.status_code == 200
    assert resp.json() == {
        "enrollment_id": "speaker-1",
        "available": False,
        "reason": "upstream_unavailable",
    }


def test_enrollment_api_accepts_raw_pcm(monkeypatch):
    monkeypatch.setattr(main_mod, "load_config", lambda: Config())
    with TestClient(main_mod.app) as client:
        resp = client.post(
            "/api/asr/enrollment",
            files={"audio": ("speaker.pcm", _pcm_bytes(), "audio/pcm")},
        )

    assert resp.status_code == 200
    assert resp.json()["duration_sec"] == 1.2


def test_enrollment_api_accepts_mp3(monkeypatch):
    monkeypatch.setattr(main_mod, "load_config", lambda: Config())
    with TestClient(main_mod.app) as client:
        resp = client.post(
            "/api/asr/enrollment",
            files={"audio": ("speaker.mp3", _mp3_bytes(), "audio/mpeg")},
        )

    assert resp.status_code == 200
    # MP3 decoder output includes codec delay/padding, so assert it lands in
    # the valid enrollment window rather than requiring sample-exact duration.
    assert 1.0 <= resp.json()["duration_sec"] <= 1.5
