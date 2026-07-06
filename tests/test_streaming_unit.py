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
import base64
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.tasks.asr as asr_task_mod  # noqa: E402
from backend.asr.k2 import asr_pb2 as k2_pb  # noqa: E402
from backend.config import SAMPLE_RATE, load_config  # noqa: E402
from backend.main import _RevalidateStaticFiles  # noqa: E402
from backend.streaming.audio_stream import (  # noqa: E402
    VadSegmentedStream,
    WholeUtteranceStream,
)
from backend.streaming.events import PartialSnapshot, PartialText, SegmentReady  # noqa: E402
from backend.streaming.k2_stream import K2SegmentedStream  # noqa: E402
from backend.streaming.protocol import (  # noqa: E402
    AstV3Protocol,
    ControlAction,
    NativeProtocol,
    PcmAction,
)
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


@pytest.mark.asyncio
async def test_static_mount_rejects_unknown_websocket(tmp_path: Path):
    static = _RevalidateStaticFiles(directory=str(tmp_path), html=True)
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "websocket.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    await static(
        {"type": "websocket", "path": "/ws/audio", "root_path": ""},
        receive,
        send,
    )

    assert sent == [{"type": "websocket.close", "code": 1008}]


def _tone_pcm_bytes(
    n_samples: int, freq: float = 440.0, sr: int = 16000, amp: float = 0.6
) -> bytes:
    t = np.arange(n_samples, dtype=np.float32) / sr
    sig = amp * np.sin(2 * np.pi * freq * t)
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


def test_vad_segmented_stream_partial_override_disables_snapshots():
    """``enable_partial=False`` must keep partials off even if cfg says yes."""
    cfg = load_config().override(enable_pseudo_stream=True)
    stream = VadSegmentedStream(enable_partial=False)
    stream.configure(cfg)
    assert stream._enable_partial is False


def test_vad_segmented_stream_partial_default_follows_cfg():
    cfg_on = load_config().override(enable_pseudo_stream=True)
    s_on = VadSegmentedStream()
    s_on.configure(cfg_on)
    assert s_on._enable_partial is True

    cfg_off = load_config().override(enable_pseudo_stream=False)
    s_off = VadSegmentedStream()
    s_off.configure(cfg_off)
    assert s_off._enable_partial is False


def test_vad_segmented_stream_configure_applies_vad_overrides():
    """Per-connection VAD overrides must reach the live VADProcessor.

    Regression: configure() previously only wired pseudo-stream knobs, so
    vad_threshold / silence_duration_ms / vad_start_frames overrides were
    silently dropped (the VAD stayed frozen to process-wide defaults).
    """
    cfg = load_config().override(
        vad_threshold=0.22, silence_duration_ms=80, vad_start_frames=3
    )
    stream = VadSegmentedStream()
    stream.configure(cfg)
    fm = stream.vad.frame_ms
    assert stream.vad.threshold == 0.22
    assert stream.vad.start_frames == 3
    assert stream.vad.silence_frames == max(1, math.ceil(80 / fm))


class _SpeakingFakeVad:
    """受控 VAD 替身:首帧即进入 speaking、process 永不切段、snapshot 返回固定
    长度。用来隔离 partial 门槛逻辑,不依赖真实 TenVad/energy 的语音判定。"""

    def __init__(self, *, snapshot_samples: int, hop_size: int = 256):
        self.hop_size = hop_size
        self.is_speaking = False
        self._snap = np.zeros(snapshot_samples, dtype=np.float32)

    def apply_config(self, cfg) -> None:  # noqa: D401 - test stub
        pass

    def process(self, frame):
        self.is_speaking = True
        return None  # 永不返回 segment -> 维持 speaking,只走 partial 路径

    def snapshot_incomplete_speech(self):
        return self._snap if self.is_speaking else None

    def flush(self):
        return None


def test_partial_uses_decoupled_floor_below_min_segment():
    """解耦核心:snapshot 时长介于 partial 门槛(200ms)与 min_segment(350ms)之间
    时仍产出 PartialSnapshot。回归点:解耦前 partial 复用 min_segment 的 min_samples,
    250ms 的 snapshot 会被 350ms 卡住、首字延后。"""
    cfg = load_config().override(
        pseudo_stream_first_partial_ms=200,
        min_segment_duration_ms=350,
        enable_pseudo_stream=True,
    )
    stream = VadSegmentedStream()
    stream.configure(cfg)
    snap_samples = int(SAMPLE_RATE * 250 / 1000)  # 250ms
    stream.vad = _SpeakingFakeVad(snapshot_samples=snap_samples)

    events = list(stream.feed(_tone_pcm_bytes(stream.vad.hop_size * 4)))
    kinds = [type(e).__name__ for e in events]
    assert "SpeechStarted" in kinds
    partials = [e for e in events if isinstance(e, PartialSnapshot)]
    assert len(partials) == 1
    assert partials[0].pcm.shape == (snap_samples,)


def test_partial_floor_still_blocks_too_short_snapshot():
    """对照:snapshot 短于 partial 门槛时不产 partial(门槛仍生效,只是被调低)。"""
    cfg = load_config().override(
        pseudo_stream_first_partial_ms=200,
        min_segment_duration_ms=350,
        enable_pseudo_stream=True,
    )
    stream = VadSegmentedStream()
    stream.configure(cfg)
    snap_samples = int(SAMPLE_RATE * 120 / 1000)  # 120ms < 200ms 门槛
    stream.vad = _SpeakingFakeVad(snapshot_samples=snap_samples)

    events = list(stream.feed(_tone_pcm_bytes(stream.vad.hop_size * 4)))
    assert not any(isinstance(e, PartialSnapshot) for e in events)


def test_k2_endpoint_emits_full_buffer_without_local_head_trim():
    stream = K2SegmentedStream()
    cfg = load_config().override(k2_idle_keep_ms=500, k2_max_segment_sec=0)
    stream.configure(cfg)
    stream._voice_gate = _FakeK2VoiceGate(has_voice=True)

    leading_silence = int(SAMPLE_RATE * 0.5)
    soft_onset = int(SAMPLE_RATE * 0.5)
    stream.feed(_silent_pcm_bytes(leading_silence))
    # Low-energy speech-like onset: old local TenVad trimming could miss this
    # and cut the segment head; k2 mode must preserve the buffered audio.
    stream.feed(_tone_pcm_bytes(soft_onset, amp=0.006))
    stream._emit_buffered_segment(is_stop_flush=False)

    ev = stream._event_q.get_nowait()
    assert isinstance(ev, SegmentReady)
    assert ev.pcm.dtype == np.float32
    assert ev.pcm.shape == (leading_silence + soft_onset,)
    assert np.allclose(ev.pcm[:leading_silence], 0.0)
    assert np.max(np.abs(ev.pcm[leading_silence:])) > 0.0
    assert ev.start_ms == 0.0
    assert ev.end_ms == pytest.approx(1000.0)


class _FakeK2VoiceGate:
    def __init__(self, *, has_voice: bool) -> None:
        self.has_voice = has_voice
        self.feed_calls: list[bytes] = []
        self.reset_calls = 0

    def feed(self, pcm_bytes: bytes) -> None:
        self.feed_calls.append(pcm_bytes)

    def reset(self) -> None:
        self.reset_calls += 1


def test_k2_silent_endpoint_without_partial_is_dropped():
    stream = K2SegmentedStream()
    stream.configure(load_config().override(k2_idle_keep_ms=500, k2_max_segment_sec=0))

    stream.feed(_silent_pcm_bytes(SAMPLE_RATE))
    stream._emit_buffered_segment(is_stop_flush=False)

    assert stream._event_q.empty()
    assert len(stream._buffer) == 0
    assert stream._buffer_start_sample == SAMPLE_RATE


def test_k2_idle_buffer_is_bounded_before_first_partial():
    stream = K2SegmentedStream()
    cfg = load_config().override(k2_idle_keep_ms=100, k2_max_segment_sec=0)
    stream.configure(cfg)

    stream.feed(_silent_pcm_bytes(SAMPLE_RATE))

    keep_bytes = int(SAMPLE_RATE * 0.1) * 2
    assert len(stream._buffer) == keep_bytes
    assert stream._buffer_start_sample == SAMPLE_RATE - int(SAMPLE_RATE * 0.1)


def test_k2_idle_buffer_preserves_fast_fed_speech():
    stream = K2SegmentedStream()
    cfg = load_config().override(
        k2_idle_keep_ms=100,
        k2_max_segment_sec=0,
        vad_start_frames=1,
        vad_threshold=0.1,
    )
    stream.configure(cfg)

    stream.feed(_tone_pcm_bytes(SAMPLE_RATE))

    assert stream._idle_seen_speech is True
    assert len(stream._buffer) == SAMPLE_RATE * 2
    assert stream._buffer_start_sample == 0


def test_k2_noisy_endpoint_without_voice_gate_evidence_is_dropped():
    stream = K2SegmentedStream()
    stream.configure(load_config().override(k2_idle_keep_ms=500, k2_max_segment_sec=0))
    stream._voice_gate = _FakeK2VoiceGate(has_voice=False)

    noise = (np.ones(SAMPLE_RATE, dtype=np.int16) * 1200).tobytes()
    stream.feed(noise)
    stream._emit_buffered_segment(is_stop_flush=False)

    assert stream._event_q.empty()
    assert len(stream._buffer) == 0
    assert stream._voice_gate.reset_calls == 1


def test_k2_force_cut_emits_segment():
    stream = K2SegmentedStream()
    cfg = load_config().override(k2_max_segment_sec=0.01, k2_idle_keep_ms=0)
    stream.configure(cfg)
    stream._voice_gate = _FakeK2VoiceGate(has_voice=True)

    stream.feed(_tone_pcm_bytes(320))

    ev = stream._event_q.get_nowait()
    assert isinstance(ev, SegmentReady)
    assert ev.pcm.shape == (320,)
    assert len(stream._buffer) == 0


def test_k2_voice_gate_can_be_disabled_for_legacy_behavior():
    stream = K2SegmentedStream()
    cfg = load_config().override(
        k2_max_segment_sec=0.01,
        k2_idle_keep_ms=0,
        k2_voice_gate_enabled=False,
    )
    stream.configure(cfg)

    stream.feed(_tone_pcm_bytes(320))

    ev = stream._event_q.get_nowait()
    assert isinstance(ev, SegmentReady)
    assert ev.pcm.shape == (320,)


class _FakeK2Partial:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeK2Event:
    def __init__(self, payload: str, *, text: str = "") -> None:
        self._payload = payload
        self.partial = _FakeK2Partial(text)

    def WhichOneof(self, _name: str) -> str:
        return self._payload


class _FakeK2Call:
    def __init__(self, events: list[_FakeK2Event]) -> None:
        self._events = events

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


@pytest.mark.asyncio
async def test_k2_partial_is_suppressed_until_voice_gate_passes():
    stream = K2SegmentedStream()
    stream.configure(load_config())
    stream._voice_gate = _FakeK2VoiceGate(has_voice=False)
    stream._call = _FakeK2Call([
        _FakeK2Event("partial", text="豪"),
        _FakeK2Event("session_ended"),
    ])

    await stream._recv_loop()

    queued = list(stream._event_q._queue)
    assert not any(isinstance(item, PartialText) for item in queued)


@pytest.mark.asyncio
async def test_k2_partial_passes_after_voice_gate_evidence():
    stream = K2SegmentedStream()
    stream.configure(load_config())
    stream._voice_gate = _FakeK2VoiceGate(has_voice=True)
    stream._call = _FakeK2Call([
        _FakeK2Event("partial", text="你好"),
        _FakeK2Event("session_ended"),
    ])

    await stream._recv_loop()

    queued = list(stream._event_q._queue)
    partials = [item for item in queued if isinstance(item, PartialText)]
    assert len(partials) == 1
    assert partials[0].text == "你好"
    assert partials[0].id == "seg-1"


@pytest.mark.asyncio
async def test_k2_request_config_matches_server_streaming_defaults():
    stream = K2SegmentedStream()
    stream.configure(load_config())

    req_iter = stream._request_iter()
    first = await req_iter.__anext__()
    await req_iter.aclose()

    cfg = first.session_config
    assert cfg.audio_format.sample_rate == SAMPLE_RATE
    assert cfg.audio_format.encoding == k2_pb.PCM_S16LE
    assert cfg.audio_format.channels == 1
    assert cfg.decoding.method == k2_pb.GREEDY_SEARCH
    assert cfg.enable_endpoint is True
    assert cfg.include_token_timestamps is False


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
            payload = {"type": "ack", "n": len(seg.pcm)}
            if seg.id:
                payload["id"] = seg.id
            return await ctx.send_json(payload)
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


class _AsyncScriptedStream:
    """Async stream fake that emits ready-to-send k2-style events."""

    def __init__(self):
        self.cfg = None
        self.started = False
        self.feed_calls: list[bytes] = []
        self.flush_calls: list[bool] = []
        self._events: asyncio.Queue = asyncio.Queue()

    def configure(self, cfg):
        self.cfg = cfg

    async def start(self):
        self.started = True

    def feed(self, pcm_bytes):
        self.feed_calls.append(pcm_bytes)
        self._events.put_nowait(PartialText(text="k2 partial", id="seg-k2-1"))
        self._events.put_nowait(
            SegmentReady(
                pcm=np.ones(400, dtype=np.float32) * 0.1,
                id="seg-k2-1",
                start_ms=10.0,
                end_ms=35.0,
            )
        )
        return []

    async def events(self):
        while True:
            ev = await self._events.get()
            if ev is None:
                break
            yield ev

    async def flush(self, *, force):
        self.flush_calls.append(force)
        self._events.put_nowait(None)
        return []

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_session_dispatches_start_pcm_segment_and_stop():
    seg = SegmentReady(pcm=np.ones(800, dtype=np.float32) * 0.1)
    stream = _ScriptedStream(feed_events=[[seg]], flush_events=[])
    engine = _RecorderEngine()
    ws = FakeWebSocket([
        {"text": '{"type":"start","format":"pcm_s16le","sample_rate_hz":16000,"channels":1,"language":"zh","hotword_pool_id":"tenant-a","hotwords":["a","b"]}'},
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
    assert session.ctx.hotword_pool_id == "tenant-a"
    assert len(engine.segments) == 1
    assert engine.segments[0].pcm.shape == (800,)
    assert engine.stop_calls == [(True, True)]
    assert stream.flush_calls and stream.flush_calls[-1] is True


@pytest.mark.asyncio
async def test_session_rejects_legacy_start_user_id():
    stream = _ScriptedStream(feed_events=[], flush_events=[])
    engine = _RecorderEngine()
    ws = FakeWebSocket([
        {"text": '{"type":"start","format":"pcm_s16le","sample_rate_hz":16000,"channels":1,"user_id":"tenant-a"}'},
    ])

    session = StreamingSession(ws, stream=stream, engine=engine)
    await session.run()
    await session.cleanup()

    assert any(
        msg.get("type") == "error" and msg.get("code") == "invalid_hotword_pool_id"
        for msg in ws.sent
    )
    assert not engine.starts


@pytest.mark.asyncio
async def test_session_dispatches_async_partial_text_and_segment():
    stream = _AsyncScriptedStream()
    engine = _RecorderEngine()
    ws = FakeWebSocket([
        {"text": '{"type":"start","sample_rate_hz":16000,"channels":1}'},
        {"bytes": _silent_pcm_bytes(160)},
        {"text": '{"type":"stop"}'},
    ])

    session = StreamingSession(ws, stream=stream, engine=engine)
    await session.run()
    await session.cleanup()

    assert stream.started is True
    assert stream.flush_calls and stream.flush_calls[-1] is True
    assert any(
        m == {"type": "partial", "text": "k2 partial", "language": "", "id": "seg-k2-1"}
        for m in ws.sent
    )
    assert any(m.get("type") == "ack" and m.get("id") == "seg-k2-1" for m in ws.sent)
    assert len(engine.segments) == 1
    assert engine.partials == []  # PartialText must not call handle_partial/vLLM.


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
async def test_emotion_engine_streaming_mode_skips_empty_fallback():
    """In segmented streaming mode, a silent session must not emit a final."""
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config()
    ctx = SessionContext(cfg=cfg, send_json=_send_json)
    engine = EmotionTaskEngine(streaming=True)
    await engine.on_start({"type": "start", "mode": "ser"}, ctx)
    await engine.on_stop(ctx, sent_any_response=False, stopped=True)
    assert sent == []


@pytest.mark.asyncio
async def test_emotion_engine_streaming_mode_emits_per_segment(monkeypatch):
    """Each VAD segment should produce its own final_emotion in streaming mode."""
    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config()
    ctx = SessionContext(cfg=cfg, language="zh", src_lang="Chinese", send_json=_send_json)

    call_count = {"n": 0}

    async def _fake_query(audio_wav_base64, *, mode="ser", base_url=None, model_name=None, timeout=None, max_tokens=None):
        call_count["n"] += 1
        return {
            "mode": mode,
            "label": "Happy" if call_count["n"] == 1 else "Sad",
            "text": "Happy" if call_count["n"] == 1 else "Sad",
            "raw_text": "",
        }

    monkeypatch.setattr("backend.tasks.emotion.query_emotion_model", _fake_query)

    engine = EmotionTaskEngine(streaming=True)
    await engine.on_start({"type": "start", "mode": "ser"}, ctx)

    seg1 = SegmentReady(pcm=np.zeros(8000, dtype=np.float32))
    seg2 = SegmentReady(pcm=np.zeros(16000, dtype=np.float32), is_stop_flush=True)
    assert await engine.handle_segment(seg1, ctx) is True
    assert await engine.handle_segment(seg2, ctx) is True
    # Streaming on_stop must NOT add a synthetic empty final since segments
    # were already sent.
    await engine.on_stop(ctx, sent_any_response=True, stopped=True)

    assert [m["type"] for m in sent] == ["final_emotion", "final_emotion"]
    assert sent[0]["label"] == "Happy"
    assert sent[1]["label"] == "Sad"
    assert sent[0]["language"] == "zh"


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
    assert sent == [
        {"type": "final", "text": "", "language": "zh", "effective_hotwords": []}
    ]


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


@pytest.mark.asyncio
async def test_asr_final_segment_voice_gate_skips_llm(monkeypatch):
    calls: list[str] = []

    async def fail_primary(*_args, **_kwargs):
        calls.append("primary")
        raise AssertionError("primary ASR should be skipped")

    async def fail_secondary(*_args, **_kwargs):
        calls.append("secondary")
        raise AssertionError("secondary ASR should be skipped")

    monkeypatch.setattr(asr_task_mod, "query_audio_model", fail_primary)
    monkeypatch.setattr(asr_task_mod, "query_audio_model_secondary", fail_secondary)
    monkeypatch.setattr(
        asr_task_mod,
        "segment_voice_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(
            accepted=False,
            reason="low_voice_ratio",
            speech_ms=0.0,
            speech_ratio=0.0,
            speech_frames=0,
            total_frames=100,
            max_prob=0.1,
            mean_prob=0.05,
            rms=0.02,
        ),
    )

    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config().override(
        enable_primary_asr=True,
        enable_secondary_asr=True,
        enable_dual_asr_fusion=True,
        asr_segment_voice_gate_enabled=True,
    )
    ctx = SessionContext(cfg=cfg, language="zh", src_lang="Chinese", send_json=_send_json)
    engine = AsrTaskEngine()

    ok = await engine.handle_segment(
        SegmentReady(pcm=np.zeros(1600, dtype=np.float32), id="noise-1"),
        ctx,
    )

    assert ok is False
    assert calls == []
    assert sent == []


@pytest.mark.asyncio
async def test_asr_final_preserves_segment_id(monkeypatch):
    async def fake_query_audio_model(*_args, **_kwargs):
        return {
            "transcription": "分句结果",
            "detected_language": "zh",
            "effective_hotwords": ["召回A", "召回B"],
        }

    async def fail_secondary(*_args, **_kwargs):
        raise AssertionError("secondary should be disabled")

    monkeypatch.setattr(asr_task_mod, "query_audio_model", fake_query_audio_model)
    monkeypatch.setattr(asr_task_mod, "query_audio_model_secondary", fail_secondary)

    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config().override(
        enable_primary_asr=True,
        enable_secondary_asr=False,
        enable_dual_asr_fusion=False,
        asr_segment_voice_gate_enabled=False,
    )
    ctx = SessionContext(cfg=cfg, language="zh", src_lang="Chinese", send_json=_send_json)
    engine = AsrTaskEngine()

    ok = await engine.handle_segment(
        SegmentReady(pcm=np.ones(1600, dtype=np.float32) * 0.05, id="seg-k2-2"),
        ctx,
    )

    assert ok is True
    assert len(sent) == 1
    assert sent[0]["type"] == "final"
    assert sent[0]["text"] == "分句结果"
    assert sent[0]["language"] == "zh"
    assert sent[0]["id"] == "seg-k2-2"
    assert sent[0]["effective_hotwords"] == ["召回A", "召回B"]
    assert sent[0]["duration_sec"] == pytest.approx(0.1)
    assert isinstance(sent[0]["audio_b64"], str) and sent[0]["audio_b64"]


@pytest.mark.asyncio
async def test_asr_partial_uses_pure_vllm_without_recall_or_hotwords(monkeypatch):
    calls: list[dict] = []

    async def fake_query_audio_model(_wav_b64, hotwords=None, **kwargs):
        calls.append({"hotwords": hotwords, **kwargs})
        return {
            "transcription": "中间结果",
            "reported_hotwords": ["不应使用"],
            "raw_text": "中间结果",
            "detected_language": "zh",
        }

    async def fail_secondary(*_args, **_kwargs):
        raise AssertionError("partial should use primary-only pure vLLM")

    monkeypatch.setattr(asr_task_mod, "query_audio_model", fake_query_audio_model)
    monkeypatch.setattr(asr_task_mod, "query_audio_model_secondary", fail_secondary)

    sent: list[dict] = []

    async def _send_json(payload):
        sent.append(payload)
        return True

    cfg = load_config().override(
        enable_primary_asr=True,
        enable_secondary_asr=True,
        enable_hotword_recall=True,
        enable_encoder_bypass=True,
    )
    ctx = SessionContext(
        cfg=cfg,
        language="zh",
        src_lang="Chinese",
        hotwords=["会话热词"],
        send_json=_send_json,
    )
    engine = AsrTaskEngine()

    await engine.handle_partial(
        PartialSnapshot(pcm=np.ones(1600, dtype=np.float32) * 0.05),
        ctx,
    )

    assert calls
    assert calls[0]["hotwords"] == []
    assert calls[0]["audio_pcm"] is None
    assert sent == [{"type": "partial", "text": "中间结果", "language": "zh"}]


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


# ---------------------------------------------------------------------------
# AST v3 wire protocol (iFlytek Tuling)
# ---------------------------------------------------------------------------


def _ast_frame(header=None, parameter=None, payload=None) -> dict:
    """Build a raw WebSocket text-frame dict carrying an AST v3 envelope."""
    env = {
        "header": header or {},
        "parameter": parameter or {},
        "payload": payload or {},
    }
    return {"text": json.dumps(env)}


def _b64_pcm(n_samples: int) -> str:
    return base64.b64encode(np.zeros(n_samples, dtype=np.int16).tobytes()).decode()


def test_native_protocol_is_identity_passthrough():
    """The default protocol must keep the historical 1:1 framing intact."""
    p = NativeProtocol()
    acts = p.decode_inbound({"text": '{"type":"start"}'})
    assert len(acts) == 1 and isinstance(acts[0], ControlAction)
    assert acts[0].ctrl == {"type": "start"}

    acts = p.decode_inbound({"bytes": b"\x00\x01"})
    assert len(acts) == 1 and isinstance(acts[0], PcmAction)
    assert acts[0].data == b"\x00\x01"

    assert p.decode_inbound({"text": "not json"}) == []

    msg = {"type": "final", "text": "x"}
    assert p.encode_outbound(msg) is msg
    assert p.encode_terminal() is None


def test_ast_v3_first_frame_synthesizes_start_with_hotwords():
    p = AstV3Protocol()
    pcm = np.zeros(160, dtype=np.int16).tobytes()
    acts = p.decode_inbound(
        _ast_frame(
            header={"traceId": "t1", "appId": "a", "bizId": "b", "status": 0},
            payload={
                "text": {"text": "挚音科技,张硕"},
                "audio": {"audio": base64.b64encode(pcm).decode()},
            },
        )
    )
    assert isinstance(acts[0], ControlAction)
    assert acts[0].ctrl["type"] == "start"
    assert acts[0].ctrl["hotwords"] == ["挚音科技", "张硕"]
    assert isinstance(acts[1], PcmAction)
    assert acts[1].data == pcm
    assert p.trace_id == "t1"
    assert p.sid.startswith("AST_")


def test_ast_v3_asr_config_injects_config_and_language():
    """parameter.asr_config -> start.config; language splits out to start.language."""
    p = AstV3Protocol()
    acts = p.decode_inbound(
        _ast_frame(
            header={"status": 0},
            parameter={
                "asr_config": {
                    "language": "en",
                    "hotword_pool_id": "tenant-a",
                    "enrollment_id": "enr-abc",
                    "vad_threshold": 0.3,
                    "enable_pseudo_stream": False,
                    "enable_role_separation": True,
                }
            },
        )
    )
    ctrl = acts[0].ctrl
    assert ctrl["type"] == "start"
    assert ctrl["language"] == "en"
    assert ctrl["hotword_pool_id"] == "tenant-a"
    assert ctrl["enrollment_id"] == "enr-abc"
    assert ctrl["config"] == {
        "vad_threshold": 0.3,
        "enable_pseudo_stream": False,
        "enable_role_separation": True,
    }


def test_ast_v3_no_asr_config_omits_config_and_language():
    """Without parameter.asr_config, the synthesized start carries neither key."""
    p = AstV3Protocol()
    acts = p.decode_inbound(_ast_frame(header={"status": 0}, parameter={"engine": {}}))
    ctrl = acts[0].ctrl
    assert "config" not in ctrl
    assert "language" not in ctrl


def test_ast_v3_asr_config_language_only():
    """asr_config with only language yields start.language and no empty config."""
    p = AstV3Protocol()
    acts = p.decode_inbound(
        _ast_frame(header={"status": 0}, parameter={"asr_config": {"language": "zh"}})
    )
    ctrl = acts[0].ctrl
    assert ctrl["language"] == "zh"
    assert "config" not in ctrl


def test_ast_v3_residlist_is_ignored_for_enrollment_id():
    """header.resIdList no longer enables target-speaker mode."""
    p = AstV3Protocol()
    acts = p.decode_inbound(
        _ast_frame(header={"traceId": "t", "status": 0, "resIdList": ["enr-abc"]})
    )
    assert acts[0].ctrl["type"] == "start"
    assert "enrollment_id" not in acts[0].ctrl


def test_ast_v3_asr_config_enrollment_id_maps_to_start():
    """parameter.asr_config.enrollment_id is the target-speaker enrollment id."""
    p = AstV3Protocol()
    acts = p.decode_inbound(
        _ast_frame(
            header={"status": 0, "resIdList": ["ignored"]},
            parameter={"asr_config": {"enrollment_id": "enr-abc"}},
        )
    )
    assert acts[0].ctrl["enrollment_id"] == "enr-abc"


def test_ast_v3_no_enrollment_when_residlist_absent_or_empty():
    for res in (None, [], [None]):
        p = AstV3Protocol()
        header = {"status": 0} if res is None else {"status": 0, "resIdList": res}
        acts = p.decode_inbound(_ast_frame(header=header))
        assert "enrollment_id" not in acts[0].ctrl, f"resIdList={res!r}"


def test_ast_v3_status_two_appends_stop():
    p = AstV3Protocol()
    p.decode_inbound(_ast_frame(header={"status": 0}))
    acts = p.decode_inbound(
        _ast_frame(header={"status": 2}, payload={"audio": {"audio": _b64_pcm(80)}})
    )
    # Trailing audio is fed before the synthesized stop.
    assert isinstance(acts[0], PcmAction)
    assert acts[-1] == ControlAction({"type": "stop"})


def test_ast_v3_status_is_parsed_leniently():
    """A client encoding status as a string must still drive the state machine."""
    p = AstV3Protocol()
    p.decode_inbound(_ast_frame(header={"status": "0"}))
    acts = p.decode_inbound(_ast_frame(header={"status": "2"}))
    assert acts[-1] == ControlAction({"type": "stop"})


def test_ast_v3_strips_leading_wav_header():
    """The reference Java SDK chunks an entire .wav; the header must be stripped."""
    from backend.audio.utils import pcm_to_wav_bytes

    p = AstV3Protocol()
    wav = pcm_to_wav_bytes(np.zeros(800, dtype=np.float32))  # 44B header + 1600B PCM
    acts = p.decode_inbound(
        _ast_frame(
            header={"status": 0},
            payload={"audio": {"audio": base64.b64encode(wav).decode()}},
        )
    )
    pcm_actions = [a for a in acts if isinstance(a, PcmAction)]
    assert pcm_actions and len(pcm_actions[0].data) == 1600


def test_ast_v3_raw_pcm_and_odd_byte_realignment():
    """Non-WAV streams pass through as PCM; odd-length chunks never lose bytes."""
    p = AstV3Protocol()
    p.decode_inbound(_ast_frame(header={"status": 0}))  # resolves start, no audio

    chunk1 = bytes(range(13))  # odd length, not RIFF
    a1 = p.decode_inbound(
        _ast_frame(
            header={"status": 1},
            payload={"audio": {"audio": base64.b64encode(chunk1).decode()}},
        )
    )
    pcm1 = b"".join(a.data for a in a1 if isinstance(a, PcmAction))
    assert len(pcm1) == 12  # trailing odd byte carried for 16-bit alignment

    chunk2 = bytes([99])
    a2 = p.decode_inbound(
        _ast_frame(
            header={"status": 1},
            payload={"audio": {"audio": base64.b64encode(chunk2).decode()}},
        )
    )
    pcm2 = b"".join(a.data for a in a2 if isinstance(a, PcmAction))
    assert pcm1 + pcm2 == chunk1 + chunk2  # no bytes lost across realignment


def test_ast_v3_final_frame_units_and_counters():
    p = AstV3Protocol()
    p.decode_inbound(_ast_frame(header={"traceId": "tt", "status": 0}))

    f1 = p.encode_outbound(
        {
            "type": "final",
            "text": "你好",
            "language": "Chinese",
            "bg_ms": 140.0,
            "ed_ms": 3230.0,
        }
    )
    assert f1["header"]["status"] == 1
    assert f1["header"]["code"] == 0
    assert f1["header"]["sid"] == p.sid
    assert f1["header"]["traceId"] == "tt"
    r = f1["payload"]["result"]
    assert r["segId"] == 0 and r["sn"] == 1
    assert r["msgtype"] == "sentence"
    # result.bg/ed in ms; vad + word offsets in 10ms frames.
    assert r["bg"] == 140 and r["ed"] == 3230
    assert r["vad"]["ws"] == [{"bg": 14, "ed": 323}]
    cw = r["ws"][0]["cw"][0]
    assert cw["w"] == "你好" and cw["lg"] == "zh"
    assert cw["wb"] == 14 and cw["we"] == 323 and cw["wp"] == "n"
    assert cw["rl"] == 0

    f2 = p.encode_outbound({"type": "final", "text": "兄弟", "language": "zh"})
    assert f2["payload"]["result"]["segId"] == 1
    assert f2["payload"]["result"]["sn"] == 2


def test_ast_v3_partial_progressive_shares_seg_id():
    p = AstV3Protocol()
    p.decode_inbound(_ast_frame(header={"status": 0}))
    part = p.encode_outbound({"type": "partial", "text": "你", "language": "zh"})
    r = part["payload"]["result"]
    assert r["msgtype"] == "Progressive"
    assert r["segId"] == 0  # same segment the upcoming sentence will carry
    cw = r["ws"][0]["cw"][0]
    assert cw["w"] == "你"
    assert "rl" not in cw
    # sentence-only fields are omitted for Progressive
    assert "sn" not in r and "vad" not in r and "bg" not in r


def test_ast_v3_suppresses_empty_final_ready_and_extract():
    p = AstV3Protocol()
    assert p.encode_outbound({"type": "final", "text": "", "language": "zh"}) is None
    assert p.encode_outbound({"type": "ready"}) is None
    assert (
        p.encode_outbound(
            {"type": "extract_hotwords_result", "request_id": "x", "hotwords": []}
        )
        is None
    )


def test_ast_v3_error_frame_nonzero_code():
    p = AstV3Protocol()
    err = p.encode_outbound({"type": "error", "message": "boom"})
    assert err["header"]["code"] != 0
    assert err["header"]["message"] == "boom"
    assert "payload" not in err  # SDK returns on code!=0 before reading payload


def test_ast_v3_terminal_is_status_two_and_idempotent():
    p = AstV3Protocol()
    p.encode_outbound({"type": "final", "text": "hi", "language": "zh"})  # advances seg
    term = p.encode_terminal()
    assert term["header"]["status"] == 2
    assert term["payload"]["result"]["ls"] is True
    assert "ws" not in term["payload"]["result"]  # getWs() == null -> SDK skips
    assert p.encode_terminal() is None  # emitted at most once


class _AstFinalEngine(BaseTaskEngine):
    """Minimal engine that forwards a segment as a final with its timing."""

    name = "ast-final"

    async def handle_segment(self, seg, ctx):
        return await ctx.send_json(
            {
                "type": "final",
                "text": "hello",
                "language": "zh",
                "bg_ms": seg.start_ms,
                "ed_ms": seg.end_ms,
            }
        )


@pytest.mark.asyncio
async def test_session_with_ast_v3_protocol_end_to_end():
    seg = SegmentReady(
        pcm=np.ones(800, dtype=np.float32) * 0.1, start_ms=100.0, end_ms=600.0
    )
    stream = _ScriptedStream(feed_events=[[seg]], flush_events=[])
    engine = _AstFinalEngine()
    ws = FakeWebSocket(
        [
            _ast_frame(
                header={"traceId": "tid", "status": 0},
                payload={"audio": {"audio": _b64_pcm(160)}},
            ),
            _ast_frame(
                header={"status": 2},
                payload={"audio": {"audio": _b64_pcm(160)}},
            ),
        ]
    )

    session = StreamingSession(
        ws, stream=stream, engine=engine, protocol=AstV3Protocol()
    )
    await session.run()
    await session.cleanup()

    # No native 'ready' leaks; every outbound frame is an AST v3 envelope.
    assert ws.sent, "expected at least the terminal frame"
    assert all("header" in m and "sid" in m["header"] for m in ws.sent)

    statuses = [m["header"]["status"] for m in ws.sent]
    assert statuses[-1] == 2  # terminal end-of-session frame is last

    sentence = next(
        m
        for m in ws.sent
        if m["header"]["status"] == 1
        and m.get("payload", {}).get("result", {}).get("msgtype") == "sentence"
    )
    assert sentence["header"]["traceId"] == "tid"
    assert sentence["payload"]["result"]["bg"] == 100
    assert sentence["payload"]["result"]["ed"] == 600
    assert sentence["payload"]["result"]["ws"][0]["cw"][0]["w"] == "hello"


@pytest.mark.asyncio
async def test_session_ast_v3_asr_config_overrides_and_whitelist():
    """End-to-end: parameter.asr_config tunes the session cfg; infra fields drop."""
    stream = _ScriptedStream(feed_events=[], flush_events=[])
    engine = _RecorderEngine()
    ws = FakeWebSocket(
        [
            _ast_frame(
                header={"status": 0},
                parameter={
                    "asr_config": {
                        "hotword_pool_id": "tenant-a",
                        "vad_threshold": 0.37,
                        "enable_role_separation": True,
                        "vllm_base_url": "http://evil:1",
                    }
                },
                payload={"audio": {"audio": _b64_pcm(160)}},
            ),
            _ast_frame(header={"status": 2}),
        ]
    )
    session = StreamingSession(
        ws, stream=stream, engine=engine, protocol=AstV3Protocol()
    )
    base_url_before = session.cfg.vllm_base_url
    await session.run()
    await session.cleanup()

    assert session.cfg.vad_threshold == 0.37  # whitelisted -> applied
    assert session.cfg.enable_role_separation is True
    assert session.cfg.vllm_base_url == base_url_before  # infra field dropped
    assert session.ctx.hotword_pool_id == "tenant-a"
    assert stream.cfg.vad_threshold == 0.37  # stream reconfigured with new cfg


@pytest.mark.asyncio
async def test_session_config_overrides_force_primary_only():
    """Endpoint-level config_overrides rebind primary and force primary-only
    (the /tuling/ast/v3 binding) at construction, before any client frame."""
    stream = _ScriptedStream(feed_events=[], flush_events=[])
    engine = _RecorderEngine()
    ws = FakeWebSocket([{"text": '{"type":"stop"}'}])
    session = StreamingSession(
        ws,
        stream=stream,
        engine=engine,
        config_overrides={
            "vllm_base_url": "http://pub:8000",
            "vllm_model_name": "Amphion-4B",
            "enable_secondary_asr": False,
        },
    )
    assert session.cfg.vllm_base_url == "http://pub:8000"
    assert session.cfg.vllm_model_name == "Amphion-4B"
    assert session.cfg.enable_secondary_asr is False
    assert session.cfg.enable_dual_asr_fusion is False  # auto-downgraded
    await session.run()
    await session.cleanup()


@pytest.mark.asyncio
async def test_session_endpoint_overrides_win_over_client():
    """A client cannot re-enable a force-disabled secondary via asr_config.

    Mirrors /tuling/ast/v3: the endpoint forces primary-only, but the AST v3
    first frame carries asr_config.enable_secondary_asr=true (a whitelisted
    field). The re-apply after override_client must keep secondary off while
    still honoring the client's other tuning (vad_threshold).
    """
    stream = _ScriptedStream(feed_events=[], flush_events=[])
    engine = _RecorderEngine()
    ws = FakeWebSocket(
        [
            _ast_frame(
                header={"status": 0},
                parameter={
                    "asr_config": {
                        "enable_secondary_asr": True,
                        "vad_threshold": 0.4,
                    }
                },
                payload={"audio": {"audio": _b64_pcm(160)}},
            ),
            _ast_frame(header={"status": 2}),
        ]
    )
    session = StreamingSession(
        ws,
        stream=stream,
        engine=engine,
        protocol=AstV3Protocol(),
        config_overrides={
            "enable_secondary_asr": False,
            "vllm_base_url": "http://pub:8000",
        },
    )
    await session.run()
    await session.cleanup()

    assert session.cfg.enable_secondary_asr is False  # endpoint lock holds
    assert session.cfg.vllm_base_url == "http://pub:8000"
    assert session.cfg.vad_threshold == 0.4  # client tuning still applied
    assert stream.cfg.enable_secondary_asr is False  # stream reconfigured too
