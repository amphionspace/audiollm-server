import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from .http_client import close_client
from .session import AudioSession
from .streaming import StreamingSession, VadSegmentedStream, WholeUtteranceStream
from .tasks import AsrTaskEngine, EmotionTaskEngine, TsAsrTaskEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_client()


app = FastAPI(title="AudioLLM Server", lifespan=lifespan)


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected (/ws/audio)")
    session = AudioSession(websocket)
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/transcribe-streaming")
async def transcribe_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info("Transcribe-streaming connected (language=%s)", language)
    session = StreamingSession(
        websocket,
        stream=VadSegmentedStream(),
        engine=AsrTaskEngine(),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/transcribe-target-streaming")
async def tsasr_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info("TS-ASR streaming connected (language=%s)", language)
    session = StreamingSession(
        websocket,
        stream=VadSegmentedStream(),
        engine=TsAsrTaskEngine(),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/emotion-streaming")
async def emotion_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info("Emotion-streaming connected (language=%s)", language)
    session = StreamingSession(
        websocket,
        stream=WholeUtteranceStream(),
        engine=EmotionTaskEngine(),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
