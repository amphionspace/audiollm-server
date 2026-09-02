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


def test_clean_stream_full_enhancement_flow(monkeypatch) -> None:
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

    with TestClient(app) as client:
        with client.websocket_connect("/asr/v1/clean-stream") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({
                "type": "session.update",
                "language": "zh",
                "cleanup": {"level": "light", "text_emotion": True},
                "hotwords": {"builtin": ["internet"], "custom": ["Amphion"]},
            })
            updated = ws.receive_json()
            assert updated["type"] == "session.updated"
            assert updated["hotwords"] == {"builtin": ["internet"], "custom_count": 1}
            ws.send_json({"type": "input_audio_buffer.append", "audio": _tone()})
            assert ws.receive_json()["type"] == "transcription.delta"
            ws.send_json({"type": "input_audio_buffer.commit", "final": True})
            emotion = ws.receive_json()
            assert emotion["type"] == "emotion.bucket"
            assert emotion["emotion"]["mode"] == "sec"
            assert ws.receive_json()["type"] == "postprocess.delta"
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


def test_clean_stream_rejects_non_boolean_emotion_option() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/asr/v1/clean-stream") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "session.update",
                "cleanup": {"level": "light", "text_emotion": "false"},
            })
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "invalid_request"
            assert error["message"] == "cleanup.text_emotion must be boolean."
