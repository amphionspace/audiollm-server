import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .audio_utils import pcm_to_wav_base64
from .config import HOP_SIZE
from .llm_client import close_client, query_audio_model
from .vad_processor import VADProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio LLM Demo")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.on_event("shutdown")
async def shutdown_event():
    await close_client()


def _generate_segment_id() -> str:
    return f"seg-{int(time.time() * 1000)}"


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    segment_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
    vad = VADProcessor()
    hotwords: list[str] = []
    stop_event = asyncio.Event()

    async def vad_task():
        """Receive audio frames (binary) and control messages (text).
        Push detected speech segments into segment_queue without waiting
        for LLM — keeps VAD fully non-blocking.
        """
        nonlocal hotwords
        try:
            while not stop_event.is_set():
                msg = await websocket.receive()

                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"]:
                    pcm = (
                        np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    for i in range(0, len(pcm), HOP_SIZE):
                        frame = pcm[i : i + HOP_SIZE]
                        if len(frame) < HOP_SIZE:
                            break
                        segment = vad.process(frame)
                        if segment is not None:
                            seg_id = _generate_segment_id()
                            try:
                                await websocket.send_json(
                                    {
                                        "type": "vad_event",
                                        "event": "segment_detected",
                                        "id": seg_id,
                                        "duration": f"{len(segment) / 16000:.1f}s",
                                    }
                                )
                            except Exception:
                                break
                            await segment_queue.put(
                                (seg_id, segment, list(hotwords))
                            )

                elif "text" in msg and msg["text"]:
                    try:
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "update_hotwords":
                            hotwords = ctrl.get("hotwords", [])
                            logger.info("Hotwords updated: %s", hotwords)
                    except json.JSONDecodeError:
                        pass

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected (vad_task)")
        finally:
            remaining = vad.flush()
            if remaining is not None and len(remaining) > 800:
                seg_id = _generate_segment_id()
                await segment_queue.put((seg_id, remaining, list(hotwords)))
            stop_event.set()
            await segment_queue.put(None)

    async def llm_task():
        """Consume speech segments from the queue, call vLLM, and send
        results back via WebSocket. Fully independent from vad_task.
        """
        while True:
            item = await segment_queue.get()
            if item is None:
                break

            seg_id, segment, hw_snapshot = item
            logger.info(
                "Processing segment %s (%.1fs, hotwords=%s)",
                seg_id,
                len(segment) / 16000,
                hw_snapshot,
            )

            try:
                await websocket.send_json(
                    {"type": "status", "id": seg_id, "status": "processing"}
                )
            except Exception:
                break

            try:
                text = await query_audio_model(
                    pcm_to_wav_base64(segment),
                    hotwords=hw_snapshot,
                )
                await websocket.send_json(
                    {"type": "response", "id": seg_id, "text": text}
                )
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.exception("LLM query failed for %s", seg_id)
                try:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "id": seg_id,
                            "message": str(e),
                        }
                    )
                except Exception:
                    break

    try:
        await asyncio.gather(vad_task(), llm_task())
    except Exception:
        logger.exception("Session error")
    finally:
        logger.info("Session ended")


# Serve frontend static files (must be mounted last)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
