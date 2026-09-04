"""Unit tests for the target-speaker enrollment feature.

Covers
------
1. The primary ASR prompt builders in ``backend/asr/client.py`` — verifies
   the exact ``messages`` structures for Amphion 4B (interleaved user text)
   and Amphion 1.7B (system text + audio-only user turn).
2. The in-memory ``EnrollmentStore`` — duration validation, tail-trim,
   TTL eviction, LRU overflow, and round-trip get/delete.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.asr.client import (  # noqa: E402
    build_audio_only_messages,
    build_primary_messages,
    detect_and_fix_repetitions,
    parse_model_output,
)
from backend.asr.enrollment import (  # noqa: E402
    EnrollmentError,
    _Store,
    decode_and_validate,
    decode_and_validate_audio_bytes,
)
from backend.audio.utils import pcm_to_wav_base64  # noqa: E402
from backend.config import SAMPLE_RATE  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_b64(seconds: float, sr: int = SAMPLE_RATE) -> str:
    n = max(1, int(round(sr * seconds)))
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.3 * np.sin(2 * np.pi * 440 * t)
    return pcm_to_wav_base64(sig.astype(np.float32), sr)


def _pcm_s16le_bytes(seconds: float, sr: int = SAMPLE_RATE) -> bytes:
    n = max(1, int(round(sr * seconds)))
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.3 * np.sin(2 * np.pi * 440 * t)
    return np.clip(sig * 32767, -32768, 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Prompt builders — must match model-specific task matrices byte-for-byte
# ---------------------------------------------------------------------------


def test_amphion_asr_task1_plain_asr():
    """Amphion 4B: ``Transcribe the following audio.`` + <audio>."""
    msgs = build_primary_messages("TARGET_B64", template="amphion_asr")
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content == [
        {"type": "text", "text": "Transcribe the following audio."},
        {"type": "input_audio", "input_audio": {"data": "TARGET_B64", "format": "wav"}},
    ]


def test_amphion_asr_task2_asr_hotwords():
    """Amphion 4B: hotwords joined with ``,`` (no spaces)."""
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["江门", "彭丽媛", "奥体中心"],
        template="amphion_asr",
    )
    text = msgs[0]["content"][0]["text"]
    assert text == "Transcribe the following audio.\nHotwords: 江门,彭丽媛,奥体中心"
    assert msgs[0]["content"][1]["input_audio"]["data"] == "TARGET_B64"


def test_amphion_asr_task5_tsasr():
    """Amphion 4B: dual text + dual audio with leading newline."""
    msgs = build_primary_messages(
        "TARGET_B64",
        enrollment_wav_base64="ENROLL_B64",
        template="amphion_asr",
    )
    content = msgs[0]["content"]
    assert content == [
        {"type": "text", "text": "Given the speaker's voice:"},
        {"type": "input_audio", "input_audio": {"data": "ENROLL_B64", "format": "wav"}},
        {
            "type": "text",
            "text": "\nTranscribe what this speaker says in the following audio.",
        },
        {"type": "input_audio", "input_audio": {"data": "TARGET_B64", "format": "wav"}},
    ]


def test_amphion_asr_task6_tsasr_hotwords():
    """Amphion 4B: second text adds ``\\nHotwords: w1,w2``."""
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["北京", "清华大学"],
        enrollment_wav_base64="ENROLL_B64",
        template="amphion_asr",
    )
    second_text = msgs[0]["content"][2]["text"]
    assert second_text == (
        "\nTranscribe what this speaker says in the following audio.\n"
        "Hotwords: 北京,清华大学"
    )


def test_amphion_asr_hotword_dedup_and_strip():
    """Hotwords are stripped + deduped while preserving order, so the
    prompt bytes never gain stray whitespace from sloppy clients."""
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["北京", "  北京  ", "上海", "上海", "", "广州"],
        template="amphion_asr",
    )
    text = msgs[0]["content"][0]["text"]
    assert text == "Transcribe the following audio.\nHotwords: 北京,上海,广州"


def test_amphion_asr_has_no_language_line():
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["北京"],
        template="amphion_asr",
    )
    assembled = "".join(
        item.get("text", "")
        for item in msgs[0]["content"]
        if item.get("type") == "text"
    )
    assert "Language:" not in assembled


def test_amphion_asr_17b_task1_plain_asr():
    msgs = build_primary_messages("TARGET_B64", template="amphion_asr_1.7b")
    assert msgs == [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "TARGET_B64", "format": "wav"},
                }
            ],
        },
    ]


def test_qwen3_asr_uses_audio_only_and_ignores_unsupported_prompt_features():
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["Amphion"],
        enrollment_wav_base64="ENROLL_B64",
        audio_embeds_b64="EMBEDS_B64",
        audio_embeds_uuid="uuid",
        template="qwen3_asr",
    )
    assert msgs == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "TARGET_B64", "format": "wav"},
                }
            ],
        }
    ]


def test_amphion_asr_17b_task2_asr_hotwords():
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["江门", "彭丽媛", "奥体中心"],
        template="amphion_asr_1.7b",
    )
    assert msgs[0] == {
        "role": "system",
        "content": "Hotwords: 江门,彭丽媛,奥体中心",
    }
    assert msgs[1]["content"][0]["input_audio"]["data"] == "TARGET_B64"


def test_amphion_asr_17b_audio_embeds_bypass():
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["江门"],
        audio_embeds_b64="EMBEDS_B64",
        audio_embeds_uuid="triton-audio-1234",
        template="amphion_asr_1.7b",
    )
    assert msgs == [
        {"role": "system", "content": "Hotwords: 江门"},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_embeds",
                    "audio_embeds": "EMBEDS_B64",
                    "uuid": "triton-audio-1234",
                }
            ],
        },
    ]


def test_amphion_asr_17b_task5_tsasr():
    msgs = build_primary_messages(
        "TARGET_B64",
        enrollment_wav_base64="ENROLL_B64",
        audio_embeds_b64="EMBEDS_B64",
        audio_embeds_uuid="triton-audio-1234",
        template="amphion_asr_1.7b",
    )
    assert msgs == [
        {
            "role": "system",
            "content": "Given the speaker's voice in the first audio.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "ENROLL_B64", "format": "wav"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": "TARGET_B64", "format": "wav"},
                },
            ],
        },
    ]


def test_amphion_asr_17b_task6_tsasr_hotwords():
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["北京", "清华大学"],
        enrollment_wav_base64="ENROLL_B64",
        template="amphion_asr_1.7b",
    )
    assert msgs[0] == {
        "role": "system",
        "content": (
            "Given the speaker's voice in the first audio.\n"
            "Hotwords: 北京,清华大学"
        ),
    }
    assert [item["input_audio"]["data"] for item in msgs[1]["content"]] == [
        "ENROLL_B64",
        "TARGET_B64",
    ]


def test_amphion_asr_17b_tsasr_audio_embeds_order():
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["北京"],
        enrollment_audio_embeds_b64="ENROLL_EMBEDS_B64",
        enrollment_audio_embeds_uuid="triton-enrollment-default-speaker",
        audio_embeds_b64="TARGET_EMBEDS_B64",
        audio_embeds_uuid="triton-audio-target",
        template="amphion_asr_1.7b",
    )

    assert msgs[0] == {
        "role": "system",
        "content": "Given the speaker's voice in the first audio.\nHotwords: 北京",
    }
    assert msgs[1]["content"] == [
        {
            "type": "audio_embeds",
            "audio_embeds": "ENROLL_EMBEDS_B64",
            "uuid": "triton-enrollment-default-speaker",
        },
        {
            "type": "audio_embeds",
            "audio_embeds": "TARGET_EMBEDS_B64",
            "uuid": "triton-audio-target",
        },
    ]


def test_unknown_primary_prompt_template_raises():
    with pytest.raises(ValueError):
        build_primary_messages("TARGET_B64", template="does_not_exist")


def test_parse_qwen3_asr_language_prefix():
    result = parse_model_output(
        "language Chinese<asr_text>你好世界",
        enable_repetition_fix=False,
    )
    assert result["transcription"] == "你好世界"
    assert result["detected_language"] == "Chinese"


def test_parse_qwen3_asr_repeated_language_prefixes():
    result = parse_model_output(
        "language Chinese<asr_text>第一段。"
        "language Chinese<asr_text>第二段。"
        "language Chinese<asr_text>第三段。",
        enable_repetition_fix=False,
    )
    assert result["transcription"] == "第一段。第二段。第三段。"
    assert result["detected_language"] == "Chinese"


def test_parse_model_output_passes_through_bare_text():
    result = parse_model_output("你好世界", enable_repetition_fix=False)
    assert result["transcription"] == "你好世界"
    assert result["detected_language"] is None


def test_detect_and_fix_repetitions_collapses_decode_loop():
    assert detect_and_fix_repetitions("哈" * 21) == "哈"
    assert detect_and_fix_repetitions("abc" * 21) == "abc"


def test_audio_only_messages_has_no_text_item():
    """The secondary (Qwen3) path is text-free single-audio prompting."""
    msgs = build_audio_only_messages("AUDIO_B64")
    assert msgs == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "AUDIO_B64", "format": "wav"},
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Enrollment store — duration validation + TTL + LRU
# ---------------------------------------------------------------------------


def test_decode_and_validate_happy_path():
    b64, dur = decode_and_validate(_wav_b64(6.0), min_sec=5.0, max_sec=10.0)
    assert isinstance(b64, str) and b64
    assert dur == pytest.approx(6.0, abs=0.05)


def test_decode_and_validate_audio_bytes_accepts_raw_pcm():
    b64, dur = decode_and_validate_audio_bytes(
        _pcm_s16le_bytes(6.0),
        audio_format="pcm",
        min_sec=5.0,
        max_sec=10.0,
    )
    assert isinstance(b64, str) and b64
    assert dur == pytest.approx(6.0, abs=0.05)


def test_decode_and_validate_audio_bytes_rejects_unknown_format():
    with pytest.raises(EnrollmentError) as exc:
        decode_and_validate_audio_bytes(
            _pcm_s16le_bytes(2.0),
            audio_format="m4a",
            min_sec=1.0,
            max_sec=8.0,
        )
    assert exc.value.code == "unsupported_format"


def test_decode_and_validate_too_short_raises():
    with pytest.raises(EnrollmentError) as exc:
        decode_and_validate(_wav_b64(4.0), min_sec=5.0, max_sec=10.0)
    assert exc.value.code == "too_short"


def test_decode_and_validate_tail_trims_when_too_long():
    """Overflows tail-trim rather than reject — matches the existing
    ASR / emotion upload convention."""
    b64, dur = decode_and_validate(_wav_b64(12.0), min_sec=5.0, max_sec=10.0)
    assert dur == pytest.approx(10.0, abs=0.05)


def test_decode_and_validate_rejects_empty():
    with pytest.raises(EnrollmentError) as exc:
        decode_and_validate("", min_sec=1.0, max_sec=8.0)
    assert exc.value.code == "empty"


def test_decode_and_validate_rejects_garbage_b64():
    with pytest.raises(EnrollmentError) as exc:
        decode_and_validate("not-actually-a-wav", min_sec=1.0, max_sec=8.0)
    assert exc.value.code == "decode_failed"


def test_store_put_get_delete_roundtrip():
    store = _Store(ttl_sec=10.0, max_entries=4)
    entry = store.put(_wav_b64(2.0), 2.0)
    assert entry.enrollment_id
    fetched = store.get(entry.enrollment_id)
    assert fetched is not None
    assert fetched.wav_base64 == entry.wav_base64
    assert store.delete(entry.enrollment_id) is True
    assert store.get(entry.enrollment_id) is None


def test_store_get_returns_none_for_missing_id():
    store = _Store(ttl_sec=10.0, max_entries=4)
    assert store.get("does-not-exist") is None
    assert store.get("") is None


def test_store_ttl_eviction_is_lazy():
    """The store doesn't run a sweeper thread; expiry is checked at
    read time. A get() past the TTL drops the entry."""
    store = _Store(ttl_sec=0.05, max_entries=4)
    entry = store.put(_wav_b64(2.0), 2.0)
    time.sleep(0.08)
    assert store.get(entry.enrollment_id) is None


def test_store_overflow_evicts_lru():
    store = _Store(ttl_sec=60.0, max_entries=3)
    ids = [store.put(_wav_b64(1.0), 1.0).enrollment_id for _ in range(3)]
    # Touch ids[1] and ids[2] so ids[0] becomes the LRU candidate.
    time.sleep(0.01)
    store.get(ids[1])
    time.sleep(0.01)
    store.get(ids[2])
    time.sleep(0.01)
    new_entry = store.put(_wav_b64(1.0), 1.0)
    assert store.get(ids[0]) is None, "oldest entry should have been evicted"
    assert store.get(ids[1]) is not None
    assert store.get(ids[2]) is not None
    assert store.get(new_entry.enrollment_id) is not None
