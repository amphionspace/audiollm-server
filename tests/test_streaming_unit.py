"""Offline unit tests for the decoupled streaming/tasks layers.

These tests exercise:
- VadSegmentedStream: PCM ingestion + carry-over + flush
- WholeUtteranceStream: full-buffer accumulation + stop-flush
- StreamingSession: protocol dispatch (start / stop / update_hotwords / PCM)
  using a fake AudioStream and TaskEngine, with no real vLLM in the loop
- EmotionTaskEngine: monkeypatched query_emotion_model -> final_emotion msg
- AsrTaskEngine.on_stop: empty-final guarantee after stop

Run with:
    .venv/bin/python -m pytest tests/test_streaming_unit.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402
from backend.streaming.audio_stream import (  # noqa: E402
    VadSegmentedStream,
    WholeUtteranceStream,
)
from backend.streaming.events import PartialSnapshot, SegmentReady  # noqa: E402
from backend.streaming.session import (  # noqa: E402
    SessionContext,
    StreamingSession,
    map_language,
)
from backend.tasks.asr import AsrTaskEngine  # noqa: E402
from backend.tasks.base import BaseTaskEngine  # noqa: E402
from backend.tasks.emotion import EmotionTaskEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silent_pcm_bytes(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _tone_pcm_bytes(n_samples: int, freq: float = 440.0, sr: int = 16000) -> bytes:
    t = np.arange(n_samples, dtype=np.float32) / sr
    sig = (0.6 * np.sin(2 * np.pi * freq * t))
    return (np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes()


class FakeWebSocket:
    """Minimal fake of starlette/fastapi WebSocket for session tests."""

    def __init__(self, scripted_messages: list[dict]):
        self._inbox = list(scripted_messages)
        self.sent: list[dict] = []
        self.closed = False

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def receive(self) -> dict:
        if not self._inbox:
            return {"type": "websocket.disconnect"}
        msg = self._inbox.pop(0)
        if "delay" in msg:
            await asyncio.sleep(msg["delay"])
            return await self.receive()
        return msg


# ---------------------------------------------------------------------------
# AudioStream tests
# ---------------------------------------------------------------------------


def test_whole_utterance_stream_accumulates_then_flushes():
    cfg = load_config()
    stream = WholeUtteranceStream()
    stream.configure(cfg)

    # feed never produces events
    assert list(stream.feed(_silent_pcm_bytes(1600))) == []
    assert list(stream.feed(_silent_pcm_bytes(800))) == []

    # flush emits one SegmentReady with all PCM concatenated
    events = list(stream.flush(force=True))
    assert len(events) == 1
    seg = events[0]
    assert isinstance(seg, SegmentReady)
    assert seg.is_stop_flush is True
    assert seg.pcm.shape == (2400,)
    assert seg.pcm.dtype == np.float32

    # subsequent flush is a no-op (buffer cleared)
    assert list(stream.flush(force=True)) == []


def test_whole_utterance_stream_empty_flush():
    cfg = load_config()
    stream = WholeUtteranceStream()
    stream.configure(cfg)
    assert list(stream.flush(force=True)) == []


def test_vad_segmented_stream_keeps_pcm_carry():
    cfg = load_config()
    stream = VadSegmentedStream()
    stream.configure(cfg)

    # Feed a small chunk that does not align with hop_size.
    hop = stream.vad.hop_size
    n = hop // 2
    events = list(stream.feed(_silent_pcm_bytes(n)))
    assert events == []
    # carry should hold those leftover samples
    assert stream._pcm_carry.shape == (n,)


def test_vad_segmented_stream_flush_returns_nothing_when_silent():
    cfg = load_config()
    stream = VadSegmentedStream()
    stream.configure(cfg)
    # Feed silence; VAD should not transition into speech
    stream.feed(_silent_pcm_bytes(16000))
    assert list(stream.flush(force=True)) == []


# ---------------------------------------------------------------------------
# Session tests with a fake engine
# ---------------------------------------------------------------------------


class _RecorderEngine(BaseTaskEngine):
    name = "recorder"

    def __init__(self):
        self.starts: list[dict] = []
        self.segments: list[SegmentReady] = []
        self.partials: list[PartialSnapshot] = []
        self.stop_calls: list[tuple[bool, bool]] = []
        self.respond = True

    async def on_start(self, ctrl, ctx):
        self.starts.append(ctrl)

    async def handle_segment(self, seg, ctx):
        self.segments.append(seg)
        if self.respond:
            return await ctx.send_json({"type": "ack", "n": len(seg.pcm)})
        return False

    async def handle_partial(self, snap, ctx):
        self.partials.append(snap)

    async def on_stop(self, ctx, *, sent_any_response, stopped):
        self.stop_calls.append((sent_any_response, stopped))


class _ScriptedStream:
    """AudioStream that emits scripted events independently of the input bytes."""

    def __init__(self, feed_events: list, flush_events: list):
        self._feed_events = list(feed_events)
        self._flush_events = list(flush_events)
        self.feed_calls: list[bytes] = []
        self.flush_calls: list[bool] = []

    def configure(self, cfg):
        self.cfg = cfg

    def feed(self, pcm_bytes):
        self.feed_calls.append(pcm_bytes)
        if not self._feed_events:
            return []
        return self._feed_events.pop(0)

    def flush(self, *, force):
        self.flush_calls.append(force)
        return list(self._flush_events) if force else []


@pytest.mark.asyncio
async def test_session_dispatches_start_pcm_segment_and_stop():
    seg = SegmentReady(pcm=np.ones(800, dtype=np.float32) * 0.1)
    stream = _ScriptedStream(feed_events=[[seg]], flush_events=[])
    engine = _RecorderEngine()
    ws = FakeWebSocket([
        {"text": '{"type":"start","format":"pcm_s16le","sample_rate_hz":16000,"channels":1,"language":"zh","hotwords":["a","b"]}'},
        {"bytes": _silent_pcm_bytes(160)},
        {"text": '{"type":"stop"}'},
    ])

    session = StreamingSession(ws, stream=stream, engine=engine)
    await session.run()
    await session.cleanup()

    sent_types = [m.get("type") for m in ws.sent]
    assert sent_types[0] == "ready"
    assert "ack" in sent_types

    assert engine.starts and engine.starts[0]["type"] == "start"
    assert len(engine.segments) == 1
    assert engine.segments[0].pcm.shape == (800,)
    assert engine.stop_calls == [(True, True)]
    assert stream.flush_calls and stream.flush_calls[-1] is True


@pytest.mark.asyncio
async def test_session_partial_dispatch_is_serialized():
    snap = PartialSnapshot(pcm=np.ones(400, dtype=np.float32) * 0.05)
    # Two PCM batches each emitting a partial; engine has slow handle_partial
    stream = _ScriptedStream(feed_events=[[snap], [snap]], flush_events=[])

    class _SlowPartialEngine(_RecorderEngine):
        async def handle_partial(self, snap_, ctx):
            await asyncio.sleep(0.05)
            self.partials.append(snap_)

    engine = _SlowPartialEngine()
    ws = FakeWebSocket([
        {"text": '{"type":"start","format":"pcm_s16le","sample_rate_hz":16000,"channels":1}'},
        {"bytes": _silent_pcm_bytes(160)},
        {"bytes": _silent_pcm_bytes(160)},
        {"text": '{"type":"stop"}'},
    ])

    session = StreamingSession(ws, stream=stream, engine=engine)
    await session.run()
    # wait for the partial task to finish if still pending
    if session._partial_task:
        await asyncio.gather(session._partial_task, return_exceptions=True)
    await session.cleanup()

    # At most one partial fired since the second arrives while the first is in
    # flight (serialized non-overlapping policy).
    assert len(engine.partials) <= 1


@pytest.mark.asyncio
async def test_session_update_hotwords_replaces_list():
    stream = _ScriptedStream(feed_events=[], flush_events=[])
    engine = _RecorderEngine()
    ws = FakeWebSocket([
        {"text": '{"type":"start","format":"pcm_s16le","sample_rate_hz":16000,"channels":1,"hotwords":["x"]}'},
        {"text": '{"type":"update_hotwords","hotwords":["y","z"],"src_lang":"en"}'},
        {"text": '{"type":"stop"}'},
    ])

    session = StreamingSession(ws, stream=stream, engine=engine)
    await session.run()
    await session.cleanup()
    assert session.ctx.hotwords == ["y", "z"]
    assert session.ctx.src_lang == "English"


# ---------------------------------------------------------------------------
# Engine tests with monkeypatched inference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emotion_engine_emits_final_emotion_ser(monkeypatch):
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config()
    ctx = SessionContext(cfg=cfg, language="zh", src_lang="Chinese", hotwords=[], send_json=_send_json)

    captured: dict = {}

    async def _fake_query(audio_wav_base64, *, mode="ser", base_url=None, model_name=None, timeout=None, max_tokens=None):
        captured["mode"] = mode
        return {
            "mode": mode,
            "label": "Happy",
            "text": "Happy",
            "raw_text": "Happy",
        }

    monkeypatch.setattr("backend.tasks.emotion.query_emotion_model", _fake_query)

    engine = EmotionTaskEngine()
    await engine.on_start({"type": "start", "mode": "ser"}, ctx)

    seg = SegmentReady(pcm=np.zeros(8000, dtype=np.float32), is_stop_flush=True)
    ok = await engine.handle_segment(seg, ctx)
    assert ok is True
    assert captured["mode"] == "ser"
    assert sent and sent[0]["type"] == "final_emotion"
    assert sent[0]["mode"] == "ser"
    assert sent[0]["label"] == "Happy"
    assert sent[0]["text"] == "Happy"
    assert sent[0]["language"] == "zh"
    assert "scores" not in sent[0]


@pytest.mark.asyncio
async def test_emotion_engine_emits_final_emotion_sec(monkeypatch):
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config()
    ctx = SessionContext(cfg=cfg, send_json=_send_json)

    captured: dict = {}
    summary = "The speaker sounds excited and cheerful, speaking at a fast pace."

    async def _fake_query(audio_wav_base64, *, mode="ser", base_url=None, model_name=None, timeout=None, max_tokens=None):
        captured["mode"] = mode
        return {
            "mode": mode,
            "label": "Happy",
            "text": summary,
            "raw_text": summary,
        }

    monkeypatch.setattr("backend.tasks.emotion.query_emotion_model", _fake_query)

    engine = EmotionTaskEngine()
    await engine.on_start({"type": "start", "mode": "sec"}, ctx)

    seg = SegmentReady(pcm=np.zeros(8000, dtype=np.float32), is_stop_flush=True)
    ok = await engine.handle_segment(seg, ctx)
    assert ok is True
    assert captured["mode"] == "sec"
    assert sent[0]["mode"] == "sec"
    assert sent[0]["text"] == summary
    assert sent[0]["label"] == "Happy"


@pytest.mark.asyncio
async def test_emotion_engine_falls_back_to_config_mode(monkeypatch):
    """When start has no mode, engine should pick Config.emotion_task_mode."""
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config().override(emotion_task_mode="sec")
    ctx = SessionContext(cfg=cfg, send_json=_send_json)

    captured: dict = {}

    async def _fake_query(audio_wav_base64, *, mode="ser", base_url=None, model_name=None, timeout=None, max_tokens=None):
        captured["mode"] = mode
        return {"mode": mode, "label": "", "text": "calm", "raw_text": "calm"}

    monkeypatch.setattr("backend.tasks.emotion.query_emotion_model", _fake_query)

    engine = EmotionTaskEngine()
    await engine.on_start({"type": "start"}, ctx)

    seg = SegmentReady(pcm=np.zeros(1600, dtype=np.float32), is_stop_flush=True)
    await engine.handle_segment(seg, ctx)
    assert captured["mode"] == "sec"
    assert sent[0]["mode"] == "sec"


@pytest.mark.asyncio
async def test_emotion_engine_on_stop_emits_empty_when_no_audio():
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config()
    ctx = SessionContext(cfg=cfg, send_json=_send_json)
    engine = EmotionTaskEngine()
    await engine.on_start({"type": "start", "mode": "ser"}, ctx)
    await engine.on_stop(ctx, sent_any_response=False, stopped=True)
    assert sent and sent[0] == {
        "type": "final_emotion",
        "mode": "ser",
        "label": "",
        "text": "",
        "duration_sec": 0.0,
    }


@pytest.mark.asyncio
async def test_asr_engine_on_stop_emits_empty_final_when_nothing_sent():
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config()
    ctx = SessionContext(cfg=cfg, language="zh", src_lang="Chinese", send_json=_send_json)
    engine = AsrTaskEngine()
    await engine.on_stop(ctx, sent_any_response=False, stopped=True)
    assert sent == [{"type": "final", "text": "", "language": "zh"}]


@pytest.mark.asyncio
async def test_asr_engine_on_stop_silent_when_response_sent():
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config()
    ctx = SessionContext(cfg=cfg, send_json=_send_json)
    engine = AsrTaskEngine()
    await engine.on_stop(ctx, sent_any_response=True, stopped=True)
    assert sent == []


@pytest.mark.asyncio
async def test_asr_engine_on_stop_silent_on_disconnect():
    """When the client closes the socket without `stop`, no empty final."""
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config()
    ctx = SessionContext(cfg=cfg, send_json=_send_json)
    engine = AsrTaskEngine()
    await engine.on_stop(ctx, sent_any_response=False, stopped=False)
    assert sent == []


# ---------------------------------------------------------------------------
# Emotion parser (aligned with AmphionASR SER/SEC outputs)
# ---------------------------------------------------------------------------


def test_emotion_parser_ser_plain_label():
    from backend.emotion.client import parse_emotion_output

    out = parse_emotion_output("Happy", mode="ser")
    assert out["mode"] == "ser"
    assert out["label"] == "Happy"
    assert out["text"] == "Happy"


def test_emotion_parser_ser_with_trailing_punctuation():
    from backend.emotion.client import parse_emotion_output

    out = parse_emotion_output("Sad.", mode="ser")
    assert out["label"] == "Sad"


def test_emotion_parser_ser_case_insensitive():
    from backend.emotion.client import parse_emotion_output

    out = parse_emotion_output("ANGRY", mode="ser")
    assert out["label"] == "Angry"


def test_emotion_parser_ser_other_complex():
    """The ``Other/Complex`` label has a slash; matching should still work."""
    from backend.emotion.client import parse_emotion_output

    out = parse_emotion_output("Other/Complex", mode="ser")
    assert out["label"] == "Other/Complex"


def test_emotion_parser_ser_handles_fenced_json_label():
    from backend.emotion.client import parse_emotion_output

    raw = '```json\n{"label": "angry"}\n```'
    out = parse_emotion_output(raw, mode="ser")
    assert out["label"] == "Angry"


def test_emotion_parser_ser_unknown_returns_empty_label():
    from backend.emotion.client import parse_emotion_output

    out = parse_emotion_output("???", mode="ser")
    assert out["label"] == ""


def test_emotion_parser_sec_returns_freeform_text():
    from backend.emotion.client import parse_emotion_output

    summary = "The speaker sounds happy and excited."
    out = parse_emotion_output(summary, mode="sec")
    assert out["mode"] == "sec"
    assert out["text"] == summary
    # SEC also surfaces a best-effort taxonomy hit harvested from the text.
    assert out["label"] == "Happy"


def test_emotion_parser_sec_strips_code_fences():
    from backend.emotion.client import parse_emotion_output

    raw = "```\nThe speaker is sad and tired.\n```"
    out = parse_emotion_output(raw, mode="sec")
    assert "sad" in out["text"].lower()
    assert out["label"] == "Sad"


def test_emotion_prompt_constants_match_amphion():
    """Defensive: catches accidental drift from the upstream training prompt."""
    from backend.emotion.prompt import (
        SEC_PROMPT,
        SER_PROMPT,
        SER_TAXONOMY,
        normalize_mode,
    )

    assert SER_PROMPT == "Classify the emotion of the following audio:"
    assert SEC_PROMPT == "Describe the emotion of the following audio:"
    assert SER_TAXONOMY == (
        "Neutral", "Happy", "Sad", "Angry",
        "Fear", "Disgust", "Surprise", "Other/Complex",
    )
    assert normalize_mode("SER") == "ser"
    assert normalize_mode("sec") == "sec"
    assert normalize_mode("???") == "ser"  # default fallback


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_map_language_codes():
    assert map_language("zh") == "Chinese"
    assert map_language("EN") == "English"
    assert map_language("English") == "English"
    assert map_language("") == "N/A"
    assert map_language("xx") == "N/A"
