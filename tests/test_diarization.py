"""Deterministic tests for diarization timeline reconciliation."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import grpc
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import SAMPLE_RATE, Config  # noqa: E402
from backend.diarization import client as client_mod  # noqa: E402
from backend.diarization.client import (  # noqa: E402
    DiarizationSession,
    DiarizationUnavailableError,
)
from backend.diarization.diarization_pb2 import (  # noqa: E402
    DiarizationEvent,
    SessionStarted,
)
from backend.diarization.turns import (  # noqa: E402
    SpeakerTurn,
    split_segment_by_speaker,
)
from backend.streaming.events import SegmentReady  # noqa: E402
from services.diarization.model import (  # noqa: E402
    ModelTurn,
    ModelUpdate,
    _probabilities_to_turns,
)
from services.diarization.server import DiarizationService  # noqa: E402


def _segment(duration_ms: int = 2000) -> SegmentReady:
    return SegmentReady(
        pcm=np.arange(duration_ms * SAMPLE_RATE // 1000, dtype=np.float32),
        id="seg-1",
        start_ms=1000.0,
        end_ms=float(1000 + duration_ms),
        is_stop_flush=True,
    )


def test_split_segment_without_finalized_turns_fails_open() -> None:
    segment = _segment()
    assert split_segment_by_speaker(segment, [], min_duration_ms=300) == [segment]


def test_split_segment_is_gap_free_and_preserves_pcm() -> None:
    segment = _segment()
    outputs = split_segment_by_speaker(
        segment,
        [SpeakerTurn(1000, 1800, 0), SpeakerTurn(1800, 3000, 1)],
        min_duration_ms=300,
    )

    assert [(out.start_ms, out.end_ms, out.speaker_index) for out in outputs] == [
        (1000.0, 1800.0, 0),
        (1800.0, 3000.0, 1),
    ]
    assert np.array_equal(np.concatenate([out.pcm for out in outputs]), segment.pcm)
    assert [out.is_stop_flush for out in outputs] == [False, True]


def test_overlap_is_assigned_to_role_with_longest_occupancy() -> None:
    outputs = split_segment_by_speaker(
        _segment(1000),
        [SpeakerTurn(1000, 1900, 1), SpeakerTurn(1400, 2000, 0)],
        min_duration_ms=0,
    )

    assert [(out.start_ms, out.end_ms, out.speaker_index) for out in outputs] == [
        (1000.0, 1900.0, 1),
        (1900.0, 2000.0, 0),
    ]


def test_short_role_flip_is_merged_without_dropping_audio() -> None:
    segment = _segment(1000)
    outputs = split_segment_by_speaker(
        segment,
        [
            SpeakerTurn(1000, 1450, 0),
            SpeakerTurn(1450, 1550, 1),
            SpeakerTurn(1550, 2000, 0),
        ],
        min_duration_ms=300,
    )

    assert [(out.start_ms, out.end_ms, out.speaker_index) for out in outputs] == [
        (1000.0, 2000.0, 0)
    ]
    assert np.array_equal(outputs[0].pcm, segment.pcm)


def test_sortformer_probabilities_preserve_overlap_and_frame_offset() -> None:
    probabilities = np.array(
        [
            [0.8, 0.1, 0.0, 0.0],
            [0.7, 0.9, 0.0, 0.0],
            [0.1, 0.8, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert _probabilities_to_turns(probabilities, frame_offset=10) == [
        ModelTurn(800, 960, 0),
        ModelTurn(880, 1040, 1),
    ]


class _UnavailableRpc(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class _ScriptedCall:
    def __init__(self, events: list[object]) -> None:
        self.events: asyncio.Queue[object] = asyncio.Queue()
        for event in events:
            self.events.put_nowait(event)
        self.cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.events.get()
        if isinstance(event, BaseException):
            raise event
        if event is StopAsyncIteration:
            raise StopAsyncIteration
        return event

    def cancel(self) -> None:
        self.cancelled = True


class _Stub:
    def __init__(self, call: _ScriptedCall) -> None:
        self.call = call

    def Diarize(self, _requests):
        return self.call


def _client_config(**overrides) -> Config:
    return Config(
        diarization_enabled=True,
        diarization_target="fake:50052",
        diarization_connect_timeout_sec=0.01,
        diarization_result_timeout_sec=0.01,
        **overrides,
    )


@pytest.mark.asyncio
async def test_connection_failure_degrades_without_leaking_task(monkeypatch) -> None:
    call = _ScriptedCall([_UnavailableRpc()])
    monkeypatch.setattr(client_mod, "get_diarization_stub", lambda _target: _Stub(call))
    session = DiarizationSession(_client_config(), session_id="failed-start")

    with pytest.raises(DiarizationUnavailableError, match="grpc_unavailable"):
        await session.start()
    await session.aclose()

    assert session.degraded_reason == "grpc_unavailable"
    assert session._recv_task is not None and session._recv_task.done()
    assert call.cancelled is True


@pytest.mark.asyncio
async def test_midstream_disconnect_degrades_remaining_session(monkeypatch) -> None:
    call = _ScriptedCall(
        [DiarizationEvent(session_started=SessionStarted())]
    )
    monkeypatch.setattr(client_mod, "get_diarization_stub", lambda _target: _Stub(call))
    session = DiarizationSession(_client_config(), session_id="midstream")

    await session.start()
    call.events.put_nowait(_UnavailableRpc())
    await session._recv_task

    assert session.available is False
    assert session.degraded_reason == "grpc_unavailable"
    assert await session.turns_for(0, 1000) == []
    await session.aclose()


@pytest.mark.asyncio
async def test_result_timeout_permanently_degrades_session() -> None:
    session = DiarizationSession(_client_config(), session_id="timeout")
    session._recv_task = asyncio.create_task(asyncio.sleep(10))

    assert await session.turns_for(0, 1000) == []
    assert session.degraded_reason == "result_timeout"

    await session.aclose()
    assert session._recv_task.done()


@pytest.mark.asyncio
async def test_client_close_cancels_pending_receive_task(monkeypatch) -> None:
    call = _ScriptedCall([DiarizationEvent(session_started=SessionStarted())])
    monkeypatch.setattr(client_mod, "get_diarization_stub", lambda _target: _Stub(call))
    session = DiarizationSession(_client_config(), session_id="early-close")

    await session.start()
    await session.aclose()

    assert session._recv_task is not None and session._recv_task.done()
    assert call.cancelled is True


@pytest.mark.asyncio
async def test_sidecar_stream_contract_start_audio_finish() -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.audio: list[bytes] = []

        def feed(self, pcm: bytes):
            self.audio.append(pcm)
            return [ModelUpdate(480, (ModelTurn(0, 480, 0),))]

        def finish(self):
            return [ModelUpdate(640, (ModelTurn(480, 640, 1),))]

    class FakeEngine:
        def __init__(self) -> None:
            self.stream = FakeStream()

        def new_stream(self):
            return self.stream

    async def requests():
        yield client_mod.pb.DiarizationRequest(
            session_config=client_mod.pb.SessionConfig(
                session_id="s1",
                sample_rate=16_000,
                channels=1,
                max_speakers=4,
            )
        )
        yield client_mod.pb.DiarizationRequest(
            audio_chunk=client_mod.pb.AudioChunk(pcm_s16le=b"\x00\x00" * 160)
        )
        yield client_mod.pb.DiarizationRequest(
            end_of_stream=client_mod.pb.EndOfStream()
        )

    engine = FakeEngine()
    service = DiarizationService(engine)  # type: ignore[arg-type]
    events = [event async for event in service.Diarize(requests(), None)]

    assert [event.WhichOneof("payload") for event in events] == [
        "session_started",
        "turns_finalized",
        "turns_finalized",
        "session_ended",
    ]
    assert engine.stream.audio == [b"\x00\x00" * 160]
    assert events[1].turns_finalized.finalized_through_ms == 480
    assert events[2].turns_finalized.turns[0].speaker_index == 1
    assert service.active_sessions == 0
