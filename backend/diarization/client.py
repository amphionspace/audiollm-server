"""Fail-open grpc.aio client for the Sortformer diarization sidecar."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import grpc

from ..config import SAMPLE_RATE, Config
from . import diarization_pb2 as pb
from . import diarization_pb2_grpc as pb_grpc
from .turns import SpeakerTurn

logger = logging.getLogger(__name__)

_channels: dict[str, grpc.aio.Channel] = {}
_REQUEST_SENTINEL = object()


class DiarizationUnavailableError(RuntimeError):
    """Raised when the optional sidecar cannot start safely."""


def get_diarization_channel(target: str) -> grpc.aio.Channel:
    target = target.strip()
    if not target:
        raise DiarizationUnavailableError("diarization_target is empty")
    channel = _channels.get(target)
    if channel is None:
        channel = grpc.aio.insecure_channel(
            target,
            options=(
                ("grpc.max_send_message_length", 8 * 1024 * 1024),
                ("grpc.max_receive_message_length", 8 * 1024 * 1024),
            ),
        )
        _channels[target] = channel
    return channel


def get_diarization_stub(target: str) -> pb_grpc.DiarizationServiceStub:
    return pb_grpc.DiarizationServiceStub(get_diarization_channel(target))


async def close_diarization_channels() -> None:
    channels = list(_channels.values())
    _channels.clear()
    for channel in channels:
        await channel.close()


async def validate_diarization_server(cfg: Config) -> None:
    if not cfg.diarization_enabled:
        return
    try:
        response = await get_diarization_stub(cfg.diarization_target).Healthz(
            pb.HealthzRequest(),
            timeout=max(0.1, float(cfg.diarization_connect_timeout_sec)),
        )
    except grpc.RpcError as exc:
        raise DiarizationUnavailableError(f"diarization Healthz failed: {exc}") from exc
    if response.status != pb.HealthzResponse.SERVING:
        raise DiarizationUnavailableError("diarization sidecar is not serving")


class DiarizationSession:
    """One sidecar stream and finalized speaker timeline per AST connection."""

    def __init__(self, cfg: Config, *, session_id: str, trace_id: str = "") -> None:
        self.cfg = cfg
        self.session_id = session_id
        self.trace_id = trace_id
        self._request_q: asyncio.Queue[object] = asyncio.Queue(maxsize=256)
        self._call = None
        self._recv_task: asyncio.Task | None = None
        self._started = asyncio.Event()
        self._updated = asyncio.Condition()
        self._turns: dict[tuple[int, int, int], SpeakerTurn] = {}
        self._finalized_through_ms = 0
        self._degraded_reason = ""
        self._closed = False
        self._eos_sent = False

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    @property
    def available(self) -> bool:
        return bool(self._recv_task) and not self._degraded_reason and not self._closed

    async def start(self) -> None:
        stub = get_diarization_stub(self.cfg.diarization_target)
        self._call = stub.Diarize(self._requests())
        self._recv_task = asyncio.create_task(self._recv_loop())
        try:
            await asyncio.wait_for(
                self._started.wait(),
                timeout=max(0.1, float(self.cfg.diarization_connect_timeout_sec)),
            )
        except asyncio.TimeoutError as exc:
            await self._degrade("connect_timeout")
            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
                await asyncio.gather(self._recv_task, return_exceptions=True)
            if self._call is not None:
                self._call.cancel()
            raise DiarizationUnavailableError("diarization session start timed out") from exc
        if self._degraded_reason:
            raise DiarizationUnavailableError(self._degraded_reason)

    async def _requests(self) -> AsyncIterator[pb.DiarizationRequest]:
        yield pb.DiarizationRequest(
            session_config=pb.SessionConfig(
                session_id=self.session_id,
                trace_id=self.trace_id,
                sample_rate=SAMPLE_RATE,
                channels=1,
                max_speakers=4,
            )
        )
        while True:
            item = await self._request_q.get()
            if item is _REQUEST_SENTINEL:
                yield pb.DiarizationRequest(end_of_stream=pb.EndOfStream())
                break
            yield pb.DiarizationRequest(audio_chunk=pb.AudioChunk(pcm_s16le=item))

    async def feed(self, pcm_s16le: bytes) -> None:
        if not pcm_s16le or not self.available:
            return
        try:
            self._request_q.put_nowait(bytes(pcm_s16le))
        except asyncio.QueueFull:
            await self._degrade("request_queue_full")

    async def turns_for(self, start_ms: float, end_ms: float) -> list[SpeakerTurn]:
        if not self.available:
            return []
        deadline = asyncio.get_running_loop().time() + max(
            0.0, float(self.cfg.diarization_result_timeout_sec)
        )
        target = int(round(end_ms))
        async with self._updated:
            while self._finalized_through_ms < target and not self._degraded_reason:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    await self._degrade_locked("result_timeout")
                    break
                try:
                    await asyncio.wait_for(self._updated.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    await self._degrade_locked("result_timeout")
                    break
        if self._degraded_reason:
            return []
        start = int(round(start_ms))
        return sorted(
            (
                turn
                for turn in self._turns.values()
                if turn.end_ms > start and turn.start_ms < target
            ),
            key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker_index),
        )

    async def finish(self) -> None:
        if self._eos_sent:
            return
        self._eos_sent = True
        if self._recv_task and not self._recv_task.done():
            await self._request_q.put(_REQUEST_SENTINEL)

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.finish()
        if self._recv_task:
            try:
                await asyncio.wait_for(
                    self._recv_task,
                    timeout=max(0.1, float(self.cfg.diarization_result_timeout_sec)),
                )
            except asyncio.TimeoutError:
                self._recv_task.cancel()
                await asyncio.gather(self._recv_task, return_exceptions=True)
        self._closed = True
        if self._call is not None:
            self._call.cancel()

    async def _recv_loop(self) -> None:
        try:
            async for event in self._call:
                which = event.WhichOneof("payload")
                if which == "session_started":
                    self._started.set()
                elif which == "turns_finalized":
                    finalized = event.turns_finalized
                    async with self._updated:
                        for raw in finalized.turns:
                            turn = SpeakerTurn(
                                start_ms=int(raw.start_ms),
                                end_ms=int(raw.end_ms),
                                speaker_index=int(raw.speaker_index),
                            )
                            self._turns[(turn.start_ms, turn.end_ms, turn.speaker_index)] = turn
                        self._finalized_through_ms = max(
                            self._finalized_through_ms,
                            int(finalized.finalized_through_ms),
                        )
                        self._updated.notify_all()
                elif which == "error":
                    await self._degrade(
                        str(event.error.code or event.error.message or "sidecar_error")
                    )
                    return
                elif which == "session_ended":
                    async with self._updated:
                        self._updated.notify_all()
                    return
            # A clean iterator EOF without SessionEnded is still an unexpected
            # mid-session close. Mark it immediately rather than making the
            # next ASR segment wait the full result timeout.
            await self._degrade("stream_closed")
        except asyncio.CancelledError:
            raise
        except grpc.RpcError as exc:
            await self._degrade(f"grpc_{exc.code().name.lower()}")
        except Exception:
            logger.exception("Diarization receive loop failed")
            await self._degrade("receive_error")
        finally:
            self._started.set()

    async def _degrade(self, reason: str) -> None:
        async with self._updated:
            await self._degrade_locked(reason)

    async def _degrade_locked(self, reason: str) -> None:
        if not self._degraded_reason:
            self._degraded_reason = reason
            logger.warning(
                "Diarization session degraded: session_id=%s trace_id=%s reason=%s",
                self.session_id,
                self.trace_id or "n/a",
                reason,
            )
        self._updated.notify_all()
