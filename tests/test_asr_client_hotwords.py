from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.asr.client as client_mod  # noqa: E402
from backend.config import Config  # noqa: E402


def test_merge_recalled_and_custom_hotwords_prefers_and_limits_custom():
    out = client_mod.merge_recalled_and_custom_hotwords(
        ["召回A", "上灯板"],
        [" 上灯板 ", "临时一", "临时二", "临时三"],
        custom_limit=2,
    )

    assert out == ["上灯板", "临时一", "召回A"]


def test_merge_recalled_and_custom_hotwords_removes_homophone_recalls():
    out = client_mod.merge_recalled_and_custom_hotwords(
        ["章硕", "张烁", "挚音科技", "张硕科技"],
        ["张硕"],
        custom_limit=8,
    )

    assert out == ["张硕", "挚音科技", "张硕科技"]


def test_merge_recalled_and_custom_hotwords_ignores_tone_for_homophones():
    out = client_mod.merge_recalled_and_custom_hotwords(
        ["妈", "麻辣"],
        ["马"],
        custom_limit=8,
    )

    assert out == ["马", "麻辣"]


def test_merge_recalled_and_custom_hotwords_does_not_filter_when_custom_limit_zero():
    out = client_mod.merge_recalled_and_custom_hotwords(
        ["章硕", "挚音科技"],
        ["张硕"],
        custom_limit=0,
    )

    assert out == ["章硕", "挚音科技"]


def test_merge_recalled_and_custom_hotwords_keeps_mixed_words_exact_only():
    out = client_mod.merge_recalled_and_custom_hotwords(
        ["阿股", "AI", "A股"],
        ["A股", "爱"],
        custom_limit=8,
    )

    assert out == ["A股", "爱", "阿股", "AI"]


@pytest.mark.asyncio
async def test_query_audio_model_merges_recalled_and_custom_hotwords(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_recall_audio(*_args, **_kwargs):
        return SimpleNamespace(
            words=["召回A", "上灯板"],
            audio_embeds_b64=None,
            uuid="triton-audio-test",
        )

    async def fake_post_chat(messages, **_kwargs):
        captured["messages"] = messages
        return {
            "transcription": "ok",
            "reported_hotwords": [],
            "raw_text": "ok",
            "detected_language": "zh",
        }

    monkeypatch.setattr(client_mod, "recall_audio", fake_recall_audio)
    monkeypatch.setattr(client_mod, "_post_chat", fake_post_chat)

    result = await client_mod.query_audio_model(
        "WAV_B64",
        hotwords=["上灯板", "临时一", "临时二"],
        audio_pcm=np.zeros(160, dtype=np.float32),
        runtime_config=Config(
            enable_hotword_recall=True,
            enable_encoder_bypass=False,
            recall_custom_hotword_limit=2,
            vllm_prompt_template="amphion_asr",
        ),
    )

    assert result["reported_hotwords"] == ["上灯板", "临时一", "召回A"]
    assert result["effective_hotwords"] == ["召回A", "上灯板"]
    messages = captured["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert content[0]["text"] == "Transcribe the following audio.\nHotwords: 上灯板,临时一,召回A"


@pytest.mark.asyncio
async def test_query_audio_model_uses_triton_enrollment_embeds(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_recall_audio(*_args, **kwargs):
        captured["recall_kwargs"] = kwargs
        return SimpleNamespace(
            words=["北京"],
            audio_embeds_b64="TARGET_EMBEDS_B64",
            enrollment_audio_embeds_b64="ENROLL_EMBEDS_B64",
            uuid="triton-audio-target",
        )

    async def fake_post_chat(messages, **_kwargs):
        captured["messages"] = messages
        return {
            "transcription": "北京",
            "reported_hotwords": [],
            "raw_text": "北京",
            "detected_language": "zh",
        }

    monkeypatch.setattr(client_mod, "recall_audio", fake_recall_audio)
    monkeypatch.setattr(client_mod, "_post_chat", fake_post_chat)

    result = await client_mod.query_audio_model(
        "TARGET_WAV_B64",
        hotwords=[],
        audio_pcm=np.zeros(160, dtype=np.float32),
        runtime_config=Config(
            enable_hotword_recall=True,
            enable_encoder_bypass=True,
            enable_triton_enrollment_store=True,
            enable_enrollment_embedding_bypass=True,
            vllm_prompt_template="amphion_asr_1.7b",
        ),
        enrollment_id="speaker-1",
        enrollment_scope_id="default",
    )

    recall_kwargs = captured["recall_kwargs"]
    assert recall_kwargs["enrollment_id"] == "speaker-1"
    assert recall_kwargs["enrollment_scope_id"] == "default"
    assert recall_kwargs["want_enrollment_audio_embeds"] is True
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["content"] == (
        "Given the speaker's voice in the first audio.\nHotwords: 北京"
    )
    assert [item["type"] for item in messages[1]["content"]] == [
        "audio_embeds",
        "audio_embeds",
    ]
    assert messages[1]["content"][0]["audio_embeds"] == "ENROLL_EMBEDS_B64"
    assert messages[1]["content"][1]["audio_embeds"] == "TARGET_EMBEDS_B64"
    assert result["enrollment_applied"] is True


@pytest.mark.asyncio
async def test_query_audio_model_marks_missing_triton_enrollment(monkeypatch):
    async def fake_recall_audio(*_args, **_kwargs):
        return SimpleNamespace(
            words=[],
            audio_embeds_b64="TARGET_EMBEDS_B64",
            enrollment_audio_embeds_b64=None,
            uuid="triton-audio-target",
        )

    async def fake_post_chat(_messages, **_kwargs):
        return {"transcription": "北京", "raw_text": "北京", "detected_language": "zh"}

    monkeypatch.setattr(client_mod, "recall_audio", fake_recall_audio)
    monkeypatch.setattr(client_mod, "_post_chat", fake_post_chat)

    result = await client_mod.query_audio_model(
        "TARGET_WAV_B64",
        audio_pcm=np.zeros(160, dtype=np.float32),
        runtime_config=Config(
            enable_hotword_recall=True,
            enable_encoder_bypass=True,
            enable_triton_enrollment_store=True,
            enable_enrollment_embedding_bypass=True,
            vllm_prompt_template="amphion_asr_1.7b",
        ),
        enrollment_id="missing-speaker",
        enrollment_scope_id="default",
    )

    assert result["enrollment_applied"] is False
    assert result["enrollment_fallback_reason"] == "not_found"


@pytest.mark.asyncio
async def test_query_audio_model_marks_enrollment_upstream_unavailable(monkeypatch):
    async def fake_recall_audio(*_args, **_kwargs):
        raise RuntimeError("recall down")

    async def fake_post_chat(_messages, **_kwargs):
        return {"transcription": "北京", "raw_text": "北京", "detected_language": "zh"}

    monkeypatch.setattr(client_mod, "recall_audio", fake_recall_audio)
    monkeypatch.setattr(client_mod, "_post_chat", fake_post_chat)

    result = await client_mod.query_audio_model(
        "TARGET_WAV_B64",
        audio_pcm=np.zeros(160, dtype=np.float32),
        runtime_config=Config(
            enable_hotword_recall=True,
            enable_encoder_bypass=True,
            enable_triton_enrollment_store=True,
            enable_enrollment_embedding_bypass=True,
            vllm_prompt_template="amphion_asr_1.7b",
        ),
        enrollment_id="speaker-1",
        enrollment_scope_id="default",
    )

    assert result["enrollment_applied"] is False
    assert result["enrollment_fallback_reason"] == "upstream_unavailable"


@pytest.mark.asyncio
async def test_query_audio_model_marks_local_enrollment_applied(monkeypatch):
    async def fake_post_chat(_messages, **_kwargs):
        return {"transcription": "北京", "raw_text": "北京", "detected_language": "zh"}

    monkeypatch.setattr(client_mod, "_post_chat", fake_post_chat)

    result = await client_mod.query_audio_model(
        "TARGET_WAV_B64",
        audio_pcm=np.zeros(160, dtype=np.float32),
        runtime_config=Config(enable_hotword_recall=False),
        enrollment_wav_base64="ENROLL_WAV_B64",
    )

    assert result["enrollment_applied"] is True
