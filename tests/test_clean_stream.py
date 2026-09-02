from __future__ import annotations

import base64

import numpy as np
from fastapi.testclient import TestClient

import backend.clean_stream as clean_stream
from backend.main import app


def _tone(seconds: float = 2.0) -> str:
    count = int(clean_stream.SAMPLE_RATE * seconds)
    times = np.arange(count, dtype=np.float32) / clean_stream.SAMPLE_RATE
    pcm = (np.sin(2 * np.pi * 440 * times) * 12_000).astype("<i2").tobytes()
    return base64.b64encode(pcm).decode("ascii")


def test_compute_delta_matches_cumulative_protocol() -> None:
    assert clean_stream.compute_delta("今天天", "今天天气") == "气"
    assert clean_stream.compute_delta("今天天汽", "今天天气很好") == "气很好"
    assert clean_stream.compute_delta("今天天气", "今天") == ""


def test_cleanup_guardrail_accepts_formatting_and_rejects_rewrites() -> None:
    assert clean_stream.evaluate_cleanup_result(
        "欢迎使用 Amphion", "欢迎使用 Amphion。", "light", ["Amphion"]
    ) == (True, "ok")
    assert clean_stream.evaluate_cleanup_result(
        "欢迎使用安费恩", "欢迎使用 Amphion。", "light", ["Amphion"]
    ) == (True, "ok")
    inserted, inserted_reason = clean_stream.evaluate_cleanup_result(
        "欢迎使用这个功能", "欢迎使用这个功能 Amphion", "light", ["Amphion"]
    )
    assert inserted is False
    assert inserted_reason.startswith("similarity:")
    accepted, reason = clean_stream.evaluate_cleanup_result(
        "今天开会讨论预算", "我们今天召开了一场重要会议并深入讨论未来战略", "light", []
    )
    assert accepted is False
    assert reason.startswith("similarity:")
    assert clean_stream.evaluate_cleanup_result("AUM 是 1200", "这是资产管理规模", "light", []) == (
        False,
        "digits_changed",
    )


def test_emotion_cleanup_may_add_emoji_but_must_preserve_punctuation() -> None:
    assert clean_stream.evaluate_cleanup_result(
        "你好，世界！",
        "你好，世界！😊",
        "light",
        [],
        preserve_punctuation=True,
    ) == (True, "ok")
    assert clean_stream.evaluate_cleanup_result(
        "你好，世界！",
        "你好世界😊",
        "light",
        [],
        preserve_punctuation=True,
    ) == (False, "punctuation_dropped")
    assert clean_stream.evaluate_cleanup_result(
        "真的吗？",
        "真的吗！😮",
        "light",
        [],
        preserve_punctuation=True,
    ) == (False, "punctuation_dropped")


def test_emotion_refine_prompt_uses_semantic_and_emotional_few_shots() -> None:
    options = clean_stream.SessionOptions(language="zh", cleanup_level="light", text_emotion=True)
    messages = clean_stream._refine_prompt(
        "继续努力！",
        options,
        {"mode": "sec", "label": "encouraging", "text": "语气鼓励、坚定"},
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "语义或交际意图" in messages[0]["content"]
    assert "最多一个" in messages[0]["content"]
    assert messages[1]["content"].startswith('{"asr_text": "加油！"')
    assert messages[2] == {"role": "assistant", "content": "加油！💪"}
    assert messages[-1]["content"].startswith('{"asr_text": "继续努力！"')


def test_clean_stream_full_enhancement_flow(monkeypatch) -> None:
    class FakeSegmentingStream:
        def __init__(self, *, enable_partial):
            assert enable_partial is True

        def configure(self, cfg):
            return None

        def feed(self, pcm):
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            return [clean_stream.SegmentReady(pcm=samples)]

        def flush(self, *, force):
            assert force is True
            return []

    async def fake_asr(pcm, options):
        assert options.language == "zh"
        return "欢迎使用 Amphion"

    async def fake_emotion(pcm, options):
        return {"mode": "sec", "label": "happy", "text": "语气轻快"}

    async def fake_refine(text, options, emotion):
        assert options.glossary
        assert emotion["mode"] == "sec"
        return "欢迎使用 Amphion。"

    monkeypatch.setattr(clean_stream, "transcribe_qwen", fake_asr)
    monkeypatch.setattr(clean_stream, "infer_emotion", fake_emotion)
    monkeypatch.setattr(clean_stream, "refine_text", fake_refine)
    monkeypatch.setattr(clean_stream, "VadSegmentedStream", FakeSegmentingStream)

    with TestClient(app) as client:
        with client.websocket_connect("/asr/v1/clean-stream") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json(
                {
                    "type": "session.update",
                    "language": "zh",
                    "cleanup": {"level": "light", "text_emotion": True},
                    "hotwords": {"builtin": ["internet"], "custom": ["Amphion"]},
                }
            )
            updated = ws.receive_json()
            assert updated["type"] == "session.updated"
            assert updated["hotwords"] == {"builtin": ["internet"], "custom_count": 1}
            ws.send_json({"type": "input_audio_buffer.append", "audio": _tone()})
            assert ws.receive_json()["type"] == "transcription.delta"
            emotion = ws.receive_json()
            assert emotion["type"] == "emotion.bucket"
            assert emotion["emotion"]["mode"] == "sec"
            assert ws.receive_json()["type"] == "postprocess.delta"
            # Segment final/refine arrives while the session is still recording.
            ws.send_json({"type": "input_audio_buffer.commit", "final": True})
            done = ws.receive_json()
            assert done["type"] == "transcription.done"
            assert done["text"] == "欢迎使用 Amphion"
            assert done["cleaned_text"] == "欢迎使用 Amphion。"
            assert done["cleanup_status"] == "completed"


def test_clean_stream_is_auth_free_and_rejects_silence() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/asr/v1/clean-stream") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({"type": "session.update", "cleanup": {"level": "off"}})
            assert ws.receive_json()["type"] == "session.updated"
            silence = base64.b64encode(bytes(clean_stream.BYTES_PER_SECOND)).decode("ascii")
            ws.send_json({"type": "input_audio_buffer.append", "audio": silence})
            ws.send_json({"type": "input_audio_buffer.commit", "final": True})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "no_speech_detected"


def test_clean_stream_has_no_sixty_second_session_limit(monkeypatch) -> None:
    class FakeSilentStream:
        def __init__(self, *, enable_partial):
            pass

        def configure(self, cfg):
            pass

        def feed(self, pcm):
            return []

        def flush(self, *, force):
            return []

    monkeypatch.setattr(clean_stream, "VadSegmentedStream", FakeSilentStream)
    long_silence = base64.b64encode(bytes(clean_stream.BYTES_PER_SECOND * 61)).decode("ascii")
    with TestClient(app) as client:
        with client.websocket_connect("/asr/v1/clean-stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "session.update", "cleanup": {"level": "off"}})
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.append", "audio": long_silence})
            ws.send_json({"type": "input_audio_buffer.commit", "final": True})
            error = ws.receive_json()
            assert error["code"] == "no_speech_detected"


def test_clean_stream_rejects_non_boolean_emotion_option() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/asr/v1/clean-stream") as ws:
            ws.receive_json()
            ws.send_json(
                {
                    "type": "session.update",
                    "cleanup": {"level": "light", "text_emotion": "false"},
                }
            )
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "invalid_request"
            assert error["message"] == "cleanup.text_emotion must be boolean."
