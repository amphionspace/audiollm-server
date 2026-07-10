"""Tests for offline long-audio transcription (segmentation + job API).

The VAD backend is replaced with a deterministic amplitude gate so the
segmentation tests don't depend on whether the optional ten-vad native
library is installed (the energy fallback's thresholds differ from TenVad's).
Everything else — VADProcessor smoothing/state machine, VadSegmentedStream
timing, the offline replay loop — runs for real.
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.asr.oneshot as oneshot_mod  # noqa: E402
import backend.asr.transcribe as transcribe_mod  # noqa: E402
import backend.config as config_mod  # noqa: E402
import backend.main as main_mod  # noqa: E402
from backend.asr.jobs import (  # noqa: E402
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_SUCCEEDED,
    JobQueueFullError,
    TranscriptionJobStore,
)
from backend.asr.transcribe import (  # noqa: E402
    float_pcm_to_i16_bytes,
    segment_pcm_offline,
    transcribe_pcm_i16,
)
from backend.audio.utils import pcm_to_wav_base64, pcm_to_wav_bytes  # noqa: E402
from backend.config import SAMPLE_RATE, load_config  # noqa: E402
from backend.jobstore import JobExecutionError  # noqa: E402
from backend.main import app  # noqa: E402


class _AmplitudeBackend:
    """Deterministic VAD backend: speech iff the frame has real amplitude."""

    def process(self, frame: np.ndarray) -> float:
        return 1.0 if float(np.abs(frame).max()) > 0.1 else 0.0


@pytest.fixture
def fake_vad(monkeypatch):
    monkeypatch.setattr(
        "backend.audio.vad.VADProcessor._create_vad_backend",
        lambda self: _AmplitudeBackend(),
    )


def _tone(sec: float) -> np.ndarray:
    n = int(SAMPLE_RATE * sec)
    return (np.sin(np.linspace(0, 2 * np.pi * 220 * sec, n)) * 0.5).astype(np.float32)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * sec), dtype=np.float32)


def _pcm_i16(*parts: np.ndarray) -> bytes:
    return float_pcm_to_i16_bytes(np.concatenate(parts))


def _wav_b64_samples(wav_b64: str) -> int:
    with wave.open(io.BytesIO(base64.b64decode(wav_b64)), "rb") as wf:
        return wf.getnframes()


def _cfg(**overrides):
    """Config with deterministic cut pauses for segmentation tests.

    Both the GLOBAL ``silence_duration_ms`` and the offline override are
    pinned to fixed test values, so the expected segment counts don't drift
    when the deployment's config.yaml is retuned (it has, e.g. global went
    350 -> 1000 ms between releases). 350 ms global + offline override off is
    the baseline these tests were written against; pass either knob in
    ``overrides`` to exercise the override path."""
    merged = {
        "silence_duration_ms": 350,
        "transcribe_silence_duration_ms": 0,
        **overrides,
    }
    return load_config().override(**merged)


# ---------------------------------------------------------------------------
# Offline segmentation
# ---------------------------------------------------------------------------


def test_segment_offline_cuts_on_silence_with_timing(fake_vad):
    pcm = _pcm_i16(
        _silence(0.5), _tone(2.0), _silence(1.0), _tone(3.0), _silence(0.5)
    )
    segs = segment_pcm_offline(pcm, _cfg(), max_segment_sec=30.0)

    assert len(segs) == 2
    first, second = segs
    # Timeline positions are approximate (pre-speech backfill, end-of-speech
    # confirmation lag) but must sit near the synthetic layout and stay
    # monotonic.
    assert first.start_ms < 1000
    assert 2000 <= first.end_ms <= 3200
    assert 2900 <= second.start_ms <= 4200
    assert second.end_ms <= 7000
    assert first.end_ms <= second.start_ms + 600  # allow backfill overlap slack
    assert first.index == 0 and second.index == 1
    # Segment PCM should cover roughly the voiced duration.
    assert 1.8 <= first.pcm.size / SAMPLE_RATE <= 3.0
    assert 2.8 <= second.pcm.size / SAMPLE_RATE <= 4.0


def test_segment_offline_force_cuts_continuous_speech(fake_vad):
    pcm = _pcm_i16(_silence(0.3), _tone(65.0), _silence(0.3))
    segs = segment_pcm_offline(pcm, _cfg(), max_segment_sec=30.0)

    # 65 s of uninterrupted speech with a 30 s cap -> two force cuts plus the
    # flushed tail.
    assert len(segs) == 3
    for seg in segs:
        # Cut lands on a 1 s feed-chunk boundary, so <= cap + 1 s.
        assert seg.pcm.size / SAMPLE_RATE <= 31.0
    ends = [seg.end_ms for seg in segs]
    assert ends == sorted(ends)
    # No audio lost at the seams: total voiced coverage stays ~65 s.
    total_sec = sum(seg.pcm.size for seg in segs) / SAMPLE_RATE
    assert total_sec >= 63.0


def test_segment_offline_silence_only_yields_nothing(fake_vad):
    segs = segment_pcm_offline(
        _pcm_i16(_silence(3.0)), _cfg(), max_segment_sec=30.0
    )
    assert segs == []


def test_transcribe_silence_override_merges_segments(fake_vad):
    # A 0.5 s pause splits speech under the global 350 ms cut pause but is
    # bridged by the offline-only 800 ms override; live endpoints keep 350 ms.
    pcm = _pcm_i16(
        _silence(0.5), _tone(2.0), _silence(0.5), _tone(2.0), _silence(1.5)
    )

    follow_global = segment_pcm_offline(pcm, _cfg(), max_segment_sec=30.0)
    assert len(follow_global) == 2

    overridden = segment_pcm_offline(
        pcm,
        _cfg(transcribe_silence_duration_ms=800),
        max_segment_sec=30.0,
    )
    assert len(overridden) == 1
    # The merged segment spans both tones (plus the bridged pause).
    assert overridden[0].pcm.size / SAMPLE_RATE >= 4.0


def test_transcribe_silence_override_does_not_leak_to_caller_cfg(fake_vad):
    cfg = _cfg(transcribe_silence_duration_ms=800)
    segment_pcm_offline(
        _pcm_i16(_silence(0.5), _tone(1.0), _silence(1.5)),
        cfg,
        max_segment_sec=30.0,
    )
    # Config is frozen; the override happens on a copy inside the pipeline.
    assert cfg.silence_duration_ms == 350


# ---------------------------------------------------------------------------
# Pipeline: retry / partial failure / assembly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_partial_failure_keeps_job_alive(fake_vad, monkeypatch):
    calls: list[int] = []

    async def fake_asr(wav_b64, **kwargs):
        n = _wav_b64_samples(wav_b64)
        calls.append(n)
        if n > int(2.5 * SAMPLE_RATE):
            return {"text": "长段成功", "language": "zh"}
        raise RuntimeError("upstream boom")

    monkeypatch.setattr(transcribe_mod, "run_oneshot_asr", fake_asr)

    pcm = _pcm_i16(
        _silence(0.5), _tone(1.5), _silence(1.0), _tone(4.0), _silence(0.5)
    )
    planned: list[int] = []
    done: list[int] = []
    result = await transcribe_pcm_i16(
        pcm,
        cfg=_cfg(),
        language="",
        on_segments_planned=planned.append,
        on_segment_done=done.append,
    )

    assert planned == [2]
    assert done[-1] == 2
    assert result["failed_segments"] == 1
    assert result["full_text"] == "长段成功"
    assert result["language"] == "zh"
    assert len(result["segments"]) == 2
    failed_entry = next(e for e in result["segments"] if "error" in e)
    assert failed_entry["text"] == ""
    assert "upstream boom" in failed_entry["error"]
    # The failed (short) segment must have been attempted twice (one retry).
    short = [n for n in calls if n <= int(2.5 * SAMPLE_RATE)]
    assert len(short) == 2


@pytest.mark.asyncio
async def test_transcribe_all_segments_failed_fails_job(fake_vad, monkeypatch):
    async def fake_asr(wav_b64, **kwargs):
        raise RuntimeError("everything is down")

    monkeypatch.setattr(transcribe_mod, "run_oneshot_asr", fake_asr)

    pcm = _pcm_i16(_silence(0.5), _tone(2.0), _silence(0.5))
    with pytest.raises(JobExecutionError):
        await transcribe_pcm_i16(pcm, cfg=_cfg())


@pytest.mark.asyncio
async def test_transcribe_empty_text_segments_dropped(fake_vad, monkeypatch):
    async def fake_asr(wav_b64, **kwargs):
        return {"text": "", "language": "zh"}

    monkeypatch.setattr(transcribe_mod, "run_oneshot_asr", fake_asr)

    pcm = _pcm_i16(_silence(0.5), _tone(2.0), _silence(0.5))
    result = await transcribe_pcm_i16(pcm, cfg=_cfg())
    # Noise that VAD passed but the model transcribed as empty: dropped, not
    # an error.
    assert result["segments"] == []
    assert result["full_text"] == ""
    assert result["failed_segments"] == 0


@pytest.mark.asyncio
async def test_transcribe_no_speech_returns_empty_result(fake_vad):
    result = await transcribe_pcm_i16(
        _pcm_i16(_silence(2.0)), cfg=_cfg()
    )
    assert result["segments"] == []
    assert result["full_text"] == ""
    assert result["duration_sec"] == pytest.approx(2.0, abs=0.01)


@pytest.mark.asyncio
async def test_oneshot_asr_removes_internal_silence_before_model(
    fake_vad,
    monkeypatch,
):
    captured: dict[str, int] = {}

    async def fake_query(wav_b64, *, audio_pcm=None, **_kwargs):
        captured["wav_samples"] = _wav_b64_samples(wav_b64)
        captured["pcm_samples"] = int(audio_pcm.size)
        return {
            "transcription": "打开遮光板",
            "detected_language": "zh",
        }

    async def fail_secondary(*_args, **_kwargs):
        raise AssertionError("secondary should be disabled")

    monkeypatch.setattr(oneshot_mod, "query_audio_model", fake_query)
    monkeypatch.setattr(oneshot_mod, "query_audio_model_secondary", fail_secondary)

    first = _tone(0.2)
    pause = _silence(0.5)
    second = _tone(0.2)
    pcm = np.concatenate([first, pause, second])
    result = await oneshot_mod.run_oneshot_asr(
        pcm_to_wav_base64(pcm),
        cfg=_cfg(
            enable_primary_asr=True,
            enable_secondary_asr=False,
            enable_dual_asr_fusion=False,
            asr_silence_removal_threshold_sec=0.5,
            vad_threshold=0.5,
            vad_smoothing_alpha=0.0,
        ),
        hotwords=[],
        language="Chinese",
        audio_pcm=pcm,
    )

    assert result["text"] == "打开遮光板"
    assert captured["wav_samples"] == int(0.4 * SAMPLE_RATE)
    assert captured["pcm_samples"] == int(0.4 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Job store semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_store_queue_full():
    store = TranscriptionJobStore()
    store.configure()
    store._queue_max = 1
    store._semaphore = asyncio.Semaphore(0)

    await store.submit(b"\x00\x00" * 100, duration_sec=0.01)
    with pytest.raises(JobQueueFullError):
        await store.submit(b"\x00\x00" * 100, duration_sec=0.01)


@pytest.mark.asyncio
async def test_job_store_releases_pcm_after_success(fake_vad, monkeypatch):
    async def fake_asr(wav_b64, **kwargs):
        return {"text": "ok", "language": "zh"}

    monkeypatch.setattr(transcribe_mod, "run_oneshot_asr", fake_asr)

    store = TranscriptionJobStore()
    store.configure()
    pcm = _pcm_i16(_silence(0.5), _tone(2.0), _silence(0.5))
    job = await store.submit(pcm, duration_sec=3.0)
    for _ in range(100):
        await asyncio.sleep(0.02)
        polled = await store.get(job.job_id)
        assert polled is not None
        if polled.status == JOB_STATUS_SUCCEEDED:
            assert polled.pcm_i16 == b""
            assert polled.result is not None
            assert polled.result["full_text"] == "ok"
            poll_dict = polled.to_poll_dict()
            assert poll_dict["progress"] == {
                "segments_total": 1,
                "segments_done": 1,
            }
            return
        if polled.status == JOB_STATUS_FAILED:
            pytest.fail(str(polled.error))
    pytest.fail("job did not complete")


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def _wav_upload_bytes(*parts: np.ndarray) -> bytes:
    return pcm_to_wav_bytes(np.concatenate(parts))


def test_transcription_http_create_poll_succeeds(fake_vad, monkeypatch):
    async def fake_asr(wav_b64, **kwargs):
        assert kwargs["hotword_pool_id"] == "tenant-a"
        return {"text": "会议第一句", "language": "zh"}

    monkeypatch.setattr(transcribe_mod, "run_oneshot_asr", fake_asr)

    # `with` keeps the TestClient's event loop alive between requests, like a
    # real server. Without it each request runs a throwaway loop, and the job
    # task — now suspended in asyncio.to_thread during segmentation — would
    # never be resumed once the POST's loop is gone.
    with TestClient(app) as client:
        wav = _wav_upload_bytes(_silence(0.5), _tone(2.0), _silence(0.5))

        create = client.post(
            "/api/asr/transcriptions",
            files={"audio": ("meeting.wav", wav, "audio/wav")},
            data={"language": "zh", "hotwords": "挚音科技", "hotword_pool_id": "tenant-a"},
        )
        assert create.status_code == 202
        body = create.json()
        assert body["status"] == JOB_STATUS_QUEUED
        assert body["poll_url"].endswith(body["job_id"])
        assert body["duration_sec"] == pytest.approx(3.0, abs=0.01)
        job_id = body["job_id"]

        for _ in range(100):
            poll = client.get(f"/api/asr/transcriptions/{job_id}")
            assert poll.status_code == 200
            data = poll.json()
            assert "progress" in data
            if data["status"] == JOB_STATUS_SUCCEEDED:
                result = data["result"]
                assert result["type"] == "transcription"
                assert result["full_text"] == "会议第一句"
                assert result["failed_segments"] == 0
                seg = result["segments"][0]
                assert seg["text"] == "会议第一句"
                assert 0 <= seg["start_ms"] < seg["end_ms"] <= 3100
                return
            if data["status"] == JOB_STATUS_FAILED:
                pytest.fail(str(data.get("error")))
            time.sleep(0.03)
        pytest.fail("poll timed out")


def test_transcription_http_rejects_overlong_audio(monkeypatch):
    cfg = load_config().override(transcribe_max_audio_sec=1.0)
    monkeypatch.setattr(main_mod, "load_transcribe_config", lambda: cfg)

    client = TestClient(app)
    wav = _wav_upload_bytes(_silence(2.0))
    resp = client.post(
        "/api/asr/transcriptions",
        files={"audio": ("long.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "split the recording" in resp.json()["detail"]


def test_transcription_http_rejects_bad_payload():
    client = TestClient(app)
    resp = client.post(
        "/api/asr/transcriptions",
        files={"audio": ("not.wav", b"definitely not a wav", "audio/wav")},
    )
    assert resp.status_code == 400
    assert "could not decode audio" in resp.json()["detail"]


def test_transcription_http_404():
    client = TestClient(app)
    resp = client.get("/api/asr/transcriptions/tr_missing123456")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_transcribe_config_defaults():
    cfg = load_config()
    assert cfg.transcribe_max_concurrent_jobs >= 1
    assert cfg.transcribe_segment_concurrency >= 1
    assert cfg.transcribe_max_segment_sec > 0
    assert cfg.transcribe_max_upload_bytes > 25 * 1024 * 1024
    assert cfg.transcribe_max_audio_sec > 3600


# ---------------------------------------------------------------------------
# rest.routes.transcribe: per-route model bindings + fusion switch
# ---------------------------------------------------------------------------


def _raw_cfg(transcribe_route: dict | None = None, **defaults_extra) -> dict:
    raw = {
        "upstreams": {
            "amphion_asr": {
                "base_url": "http://amphion:8009",
                "model_name": "AmphionASR-4.3B",
                "timeout": 120,
            },
            "qwen_asr": {
                "base_url": "http://qwen:8001",
                "model_name": "Qwen3-ASR",
                "timeout": 60,
            },
        },
        "defaults": {"asr": {"enable_secondary_asr": True,
                             "enable_dual_asr_fusion": False, **defaults_extra}},
        "rest": {"upstreams": {"primary": "amphion_asr", "secondary": "qwen_asr"}},
    }
    if transcribe_route is not None:
        raw["rest"]["routes"] = {"transcribe": transcribe_route}
    return raw


def test_rest_transcribe_absent_follows_global_bindings():
    parsed = config_mod._parse(_raw_cfg())
    assert parsed.transcribe_config == parsed.default_config
    assert parsed.transcribe_config.vllm_base_url == "http://amphion:8009"


def test_rest_transcribe_overrides_primary_model_only_for_transcription():
    parsed = config_mod._parse(
        _raw_cfg({"upstreams": {"primary": "qwen_asr"}})
    )
    assert parsed.transcribe_config.vllm_base_url == "http://qwen:8001"
    assert parsed.transcribe_config.vllm_model_name == "Qwen3-ASR"
    assert parsed.transcribe_config.asr_request_timeout == 60
    # The shared REST view is untouched.
    assert parsed.default_config.vllm_base_url == "http://amphion:8009"


def test_rest_transcribe_fusion_switch_is_route_scoped():
    parsed = config_mod._parse(_raw_cfg({"enable_dual_asr_fusion": True}))
    assert parsed.transcribe_config.enable_dual_asr_fusion is True
    assert parsed.default_config.enable_dual_asr_fusion is False


def test_rest_transcribe_fusion_downgrades_without_secondary():
    parsed = config_mod._parse(
        _raw_cfg({"enable_dual_asr_fusion": True}, enable_secondary_asr=False)
    )
    # Config.__post_init__ invariant: fusion requires a secondary decoder.
    assert parsed.transcribe_config.enable_dual_asr_fusion is False


def test_rest_transcribe_rejects_unknown_keys_roles_and_upstreams():
    with pytest.raises(ValueError, match="unknown keys"):
        config_mod._parse(_raw_cfg({"polcy": {}}))
    with pytest.raises(ValueError, match="unknown role"):
        config_mod._parse(_raw_cfg({"upstreams": {"emotion": "amphion_asr"}}))
    with pytest.raises(ValueError, match="unknown upstream"):
        config_mod._parse(_raw_cfg({"upstreams": {"primary": "nope"}}))


def test_rest_rejects_unknown_route_and_legacy_flat_key():
    # An unknown route name fails loudly...
    raw = _raw_cfg()
    raw["rest"]["routes"] = {"transcrbe": {"upstreams": {"primary": "qwen_asr"}}}
    with pytest.raises(ValueError, match="unknown routes"):
        config_mod._parse(raw)

    # ...as does the pre-restructure flat `rest.transcribe` key, so a stale
    # config can't silently fall back to the shared bindings.
    raw = _raw_cfg()
    raw["rest"]["transcribe"] = {"upstreams": {"primary": "qwen_asr"}}
    with pytest.raises(ValueError, match="rest: unknown keys"):
        config_mod._parse(raw)
