from __future__ import annotations

import io
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi import UploadFile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.asr.oneshot as oneshot_mod  # noqa: E402
import backend.main as main_mod  # noqa: E402
from backend.config import Config  # noqa: E402
from backend.text_cleanup.client import (  # noqa: E402
    TextCleanupConfigError,
    clean_asr_text,
)


def _wav_bytes(duration_sec: float = 0.2) -> bytes:
    samples = np.zeros(int(16000 * duration_sec), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _mp3_bytes(duration_sec: float = 0.2) -> bytes:
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
        input=_wav_bytes(duration_sec),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg mp3 encode failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


@pytest.mark.asyncio
async def test_audio_analyze_returns_asr_cleanup_and_emotion(monkeypatch):
    cfg = Config(
        enable_primary_asr=True,
        enable_secondary_asr=False,
        text_cleanup_api_key="test-key",
    )
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)

    async def fake_asr(*args, **kwargs):
        assert kwargs["hotwords"] == ["挚音科技", "张硕"]
        assert kwargs["hotword_pool_id"] == "tenant-a"
        return {
            "transcription": "挚音科技 今天 发布 新 功能",
            "reported_hotwords": ["挚音科技"],
            "raw_text": "Transcription: 挚音科技 今天 发布 新 功能",
            "detected_language": "zh",
        }

    async def fake_emotion(*args, **kwargs):
        mode = kwargs["mode"]
        if mode == "sec":
            return {
                "mode": "sec",
                "label": "Happy",
                "text": "The speaker sounds upbeat.",
                "raw_text": "The speaker sounds upbeat.",
            }
        return {
            "mode": "ser",
            "label": "Neutral",
            "text": "Neutral",
            "raw_text": "Neutral",
        }

    async def fake_cleanup(asr_text, **kwargs):
        assert asr_text == "挚音科技 今天 发布 新 功能"
        assert kwargs["hotwords"] == []
        return {
            "text": "挚音科技今天发布新功能。",
            "raw_text": '{"cleaned_text":"挚音科技今天发布新功能。"}',
            "model": "qwen2.5-32b-instruct",
        }

    # The one-shot dual-ASR orchestration lives in backend.asr.oneshot
    # (shared by upload / analyze / transcription jobs); patch it there.
    monkeypatch.setattr(oneshot_mod, "query_audio_model", fake_asr)
    monkeypatch.setattr(main_mod, "query_emotion_model", fake_emotion)
    monkeypatch.setattr(main_mod, "clean_asr_text", fake_cleanup)

    upload = UploadFile(file=io.BytesIO(_wav_bytes()), filename="sample.wav")
    result = await main_mod.audio_analyze(
        audio=upload,
        language="zh",
        hotwords="挚音科技,张硕",
        hotword_pool_id="tenant-a",
    )

    assert result["type"] == "audio_analysis"
    assert result["hotwords"] == ["挚音科技", "张硕"]
    assert result["asr"]["text"] == "挚音科技 今天 发布 新 功能"
    assert "raw_text" not in result["asr"]
    assert "primary" not in result["asr"]
    assert "secondary" not in result["asr"]
    assert "fusion" not in result["asr"]
    assert result["cleaned_asr"]["text"] == "挚音科技今天发布新功能。"
    assert "raw_text" not in result["cleaned_asr"]
    assert "model" not in result["cleaned_asr"]
    assert result["emotion"]["mode"] == "both"
    assert result["emotion"]["ser"]["label"] == "Neutral"
    assert "raw_text" not in result["emotion"]["ser"]
    assert result["emotion"]["sec"]["text"] == "The speaker sounds upbeat."
    assert "raw_text" not in result["emotion"]["sec"]


@pytest.mark.asyncio
async def test_asr_upload_passes_hotword_pool_id(monkeypatch):
    cfg = Config(enable_primary_asr=True, enable_secondary_asr=False)
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)

    async def fake_asr(*args, **kwargs):
        assert kwargs["hotword_pool_id"] == "tenant-a"
        return {
            "transcription": "上传结果",
            "reported_hotwords": [],
            "effective_hotwords": ["召回上传"],
            "raw_text": "上传结果",
            "detected_language": "zh",
        }

    monkeypatch.setattr(oneshot_mod, "query_audio_model", fake_asr)

    upload = UploadFile(file=io.BytesIO(_wav_bytes()), filename="sample.wav")
    result = await main_mod.asr_upload(
        audio=upload,
        language="zh",
        hotwords="",
        hotword_pool_id="tenant-a",
    )

    assert result["type"] == "final"
    assert result["text"] == "上传结果"
    assert result["effective_hotwords"] == ["召回上传"]


@pytest.mark.asyncio
async def test_asr_upload_accepts_mp3(monkeypatch):
    cfg = Config(enable_primary_asr=True, enable_secondary_asr=False)
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)

    async def fake_asr(*args, **kwargs):
        assert kwargs["audio_pcm"] is not None
        assert kwargs["audio_pcm"].size > 0
        return {
            "transcription": "mp3上传结果",
            "reported_hotwords": [],
            "effective_hotwords": [],
            "raw_text": "mp3上传结果",
            "detected_language": "zh",
        }

    monkeypatch.setattr(oneshot_mod, "query_audio_model", fake_asr)

    upload = UploadFile(file=io.BytesIO(_mp3_bytes()), filename="sample.mp3")
    result = await main_mod.asr_upload(
        audio=upload,
        language="zh",
        hotwords="",
        enrollment_id="",
        hotword_pool_id="",
    )

    assert result["type"] == "final"
    assert result["text"] == "mp3上传结果"
    assert result["duration_sec"] > 0


@pytest.mark.asyncio
async def test_clean_asr_text_requires_api_key(monkeypatch):
    monkeypatch.delenv("MISSING_DASHSCOPE_KEY", raising=False)
    cfg = Config(
        text_cleanup_api_key_env="MISSING_DASHSCOPE_KEY",
        text_cleanup_api_key="",
    )

    with pytest.raises(TextCleanupConfigError):
        await clean_asr_text("hello", cfg=cfg)
