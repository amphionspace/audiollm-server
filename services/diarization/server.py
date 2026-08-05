"""grpc.aio server for the optional Streaming Sortformer sidecar."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

import grpc

from backend.diarization import diarization_pb2 as pb
from backend.diarization import diarization_pb2_grpc as pb_grpc

from .model import (
    MAX_SPEAKERS,
    MODEL_REPO,
    InsufficientGpuMemoryError,
    SortformerEngine,
)

logger = logging.getLogger(__name__)


class DiarizationService(pb_grpc.DiarizationServiceServicer):
    def __init__(self, engine: SortformerEngine) -> None:
        self.engine = engine
        self.active_sessions = 0

    async def Healthz(self, request, context):
        return pb.HealthzResponse(
            status=pb.HealthzResponse.SERVING,
            active_sessions=self.active_sessions,
            model_name=MODEL_REPO,
        )

    async def Diarize(self, request_iterator, context):
        stream = None
        started = False
        self.active_sessions += 1
        try:
            async for request in request_iterator:
                which = request.WhichOneof("payload")
                if which == "session_config":
                    cfg = request.session_config
                    if started:
                        yield _error("duplicate_start", "session_config must be first")
                        return
                    if cfg.sample_rate != 16_000 or cfg.channels != 1:
                        yield _error("invalid_audio", "requires 16 kHz mono PCM_S16LE")
                        return
                    if cfg.max_speakers < 1 or cfg.max_speakers > MAX_SPEAKERS:
                        yield _error(
                            "invalid_speakers",
                            f"max_speakers must be in [1, {MAX_SPEAKERS}]",
                        )
                        return
                    stream = self.engine.new_stream()
                    started = True
                    yield pb.DiarizationEvent(session_started=pb.SessionStarted())
                elif which == "audio_chunk":
                    if not started or stream is None:
                        yield _error("missing_start", "session_config must be first")
                        return
                    updates = await asyncio.to_thread(stream.feed, request.audio_chunk.pcm_s16le)
                    for update in updates:
                        yield _event_from_update(update)
                elif which == "end_of_stream":
                    if stream is not None:
                        updates = await asyncio.to_thread(stream.finish)
                        for update in updates:
                            yield _event_from_update(update)
                    yield pb.DiarizationEvent(session_ended=pb.SessionEnded())
                    return
            if stream is not None:
                updates = await asyncio.to_thread(stream.finish)
                for update in updates:
                    yield _event_from_update(update)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Diarization stream failed")
            yield _error("inference_failed", str(exc))
        finally:
            self.active_sessions -= 1


def _event_from_update(update) -> pb.DiarizationEvent:
    return pb.DiarizationEvent(
        turns_finalized=pb.TurnsFinalized(
            finalized_through_ms=update.finalized_through_ms,
            turns=[
                pb.SpeakerTurn(
                    start_ms=turn.start_ms,
                    end_ms=turn.end_ms,
                    speaker_index=turn.speaker_index,
                )
                for turn in update.turns
            ],
        )
    )


def _error(code: str, message: str) -> pb.DiarizationEvent:
    return pb.DiarizationEvent(error=pb.DiarizationError(code=code, message=message))


async def serve(host: str, port: int) -> None:
    model_path = os.getenv("DIARIZATION_MODEL_PATH", "").strip()
    device = os.getenv("DIARIZATION_DEVICE", "cuda").strip() or "cuda"
    engine = await asyncio.to_thread(SortformerEngine, model_path, device=device)
    server = grpc.aio.server(
        options=(
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
        )
    )
    pb_grpc.add_DiarizationServiceServicer_to_server(DiarizationService(engine), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info("Diarization sidecar listening on %s:%d", host, port)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    try:
        await server.stop(grace=5)
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(serve(args.host, args.port))
    except InsufficientGpuMemoryError as exc:
        # EX_CONFIG lets systemd distinguish the intentional GPU safety stop
        # from a transient crash that should still be restarted.
        logger.error("Diarization sidecar stopped by GPU safety gate: %s", exc)
        raise SystemExit(78) from exc


if __name__ == "__main__":
    main()
