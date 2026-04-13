import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from .asr_streaming_session import ASRStreamingSession
from .http_client import close_client
from .session import AudioSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio LLM Demo")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.on_event("shutdown")
async def shutdown_event():
    await close_client()


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    session = AudioSession(websocket)
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/transcribe-streaming")
async def transcribe_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info("Transcribe-streaming connected (language=%s)", language)
    session = ASRStreamingSession(websocket, language)
    try:
        await session.run()
    finally:
        await session.cleanup()


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
