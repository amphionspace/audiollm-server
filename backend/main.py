import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles

from .asr.client import query_audio_model, query_audio_model_secondary
from .asr.fusion import choose_fused_result
from .audio.utils import wav_base64_to_pcm_16k_mono
from .config import SAMPLE_RATE, load_config
from .emotion.client import query_emotion_model
from .emotion.prompt import normalize_mode
from .http_client import close_client
from .session import AudioSession
from .streaming import StreamingSession, VadSegmentedStream, WholeUtteranceStream
from .tasks import AsrTaskEngine, EmotionTaskEngine, TsAsrTaskEngine
from .tsasr.client import query_tsasr_model
from .tsasr.enrollment import EnrollmentError, decode_enrollment

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


@app.websocket("/emotion-segmented-streaming")
async def emotion_segmented_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info(
        "Emotion-segmented-streaming connected (language=%s)", language
    )
    session = StreamingSession(
        websocket,
        # Emotion has no partial output, so disable VAD's snapshot bookkeeping
        # regardless of the global pseudo-stream toggle.
        stream=VadSegmentedStream(enable_partial=False),
        engine=EmotionTaskEngine(streaming=True),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


# ---------------------------------------------------------------------------
# One-shot upload endpoints
# ---------------------------------------------------------------------------
# The /api/* routes power the "Upload audio file" buttons in the demos.
# They deliberately bypass the WebSocket/VAD pipeline: the frontend hands us
# a fully-decoded 16 kHz mono WAV (produced via the browser's Web Audio API),
# and we forward the bytes to the same vLLM endpoints the streaming engines
# call. This keeps the upload flow as "send the whole clip, get one final
# result" — no chunking, no VAD segmentation, no partials.
#
# All caps that the streaming pipeline normally enforces server-side
# (emotion 20s tail, tsasr 30s tail, enrollment 1-5s VAD trim) are still
# applied here so a malicious or buggy client can't bypass them by switching
# from WS to REST.

# Hard cap on multipart upload bytes. ~16-bit / 16 kHz mono WAV at 60 s is
# ~1.9 MB; this 25 MB ceiling is generous for any clip the model would
# realistically be asked to handle.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Server-side trim caps (mirror the streaming pipeline's behaviour so REST
# and WS produce identical model inputs for the same recording).
_ASR_MAX_SECONDS = 60.0


def _parse_csv(raw: str | None) -> list[str]:
    """Parse a ``"a,b ,c"`` form field into a clean string list."""
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


async def _read_audio_bytes(audio: UploadFile) -> bytes:
    """Read a multipart audio upload, enforcing the global byte cap.

    UploadFile.read with no argument loads into memory; the size check is
    primarily a guard against accidental huge uploads, not a streaming
    safeguard (we need the full payload for vLLM anyway).
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio file is empty")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio file exceeds {_MAX_UPLOAD_BYTES} bytes",
        )
    return raw


def _wav_to_pcm_capped(raw: bytes, max_seconds: float) -> tuple[bytes, float]:
    """Decode a WAV blob to 16 kHz mono and tail-trim to ``max_seconds``.

    Returns (re_encoded_wav_bytes, duration_sec). When no trim is needed the
    re-encoded WAV is byte-equivalent to ``pcm_to_wav_base64(pcm)``.
    """
    import io
    import wave

    import numpy as np

    from .audio.utils import pcm_to_wav_base64

    wav_b64 = base64.b64encode(raw).decode("ascii")
    try:
        pcm = wav_base64_to_pcm_16k_mono(wav_b64)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc
    if pcm.size == 0:
        raise HTTPException(status_code=400, detail="audio decoded to empty PCM")
    duration = pcm.size / SAMPLE_RATE
    if max_seconds > 0 and duration > max_seconds:
        # Match streaming engines: keep the trailing window. Emotion picks
        # the tail because the most recent emotion is what users care about;
        # we use the same convention for ASR/TS-ASR for consistency.
        keep = int(SAMPLE_RATE * max_seconds)
        pcm = pcm[-keep:]
        duration = pcm.size / SAMPLE_RATE
    new_b64 = pcm_to_wav_base64(pcm.astype(np.float32, copy=False))
    new_bytes = base64.b64decode(new_b64)
    # Sanity: re-encoded WAV should still parse.
    with wave.open(io.BytesIO(new_bytes), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
    return new_bytes, duration


@app.post("/api/asr/upload")
async def asr_upload(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
):
    """One-shot ASR over an uploaded clip.

    Mirrors :class:`AsrTaskEngine.handle_segment` but operates on the entire
    clip in a single dual-model call (no VAD segmentation, no partials).
    Returns the same fields the streaming ``final`` event carries.
    """
    raw = await _read_audio_bytes(audio)
    wav_bytes, duration_sec = _wav_to_pcm_capped(raw, _ASR_MAX_SECONDS)
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    cfg = load_config()
    hw_list = _parse_csv(hotwords)

    primary_task = None
    secondary_task = None
    if cfg.enable_primary_asr:
        primary_task = asyncio.create_task(
            asyncio.wait_for(
                query_audio_model(
                    wav_b64,
                    hotwords=hw_list,
                    src_lang=language or "N/A",
                    base_url=cfg.vllm_base_url,
                    model_name=cfg.vllm_model_name,
                    timeout=cfg.asr_request_timeout,
                ),
                timeout=cfg.primary_asr_timeout,
            )
        )
    if cfg.enable_secondary_asr:
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(
                wav_b64,
                hotwords=hw_list,
                base_url=cfg.secondary_vllm_base_url,
                model_name=cfg.secondary_vllm_model_name,
                timeout=cfg.asr_request_timeout,
            )
        )

    primary_res: object | None = None
    secondary_res: object | None = None
    if primary_task is not None:
        try:
            primary_res = await primary_task
        except Exception as err:  # noqa: BLE001 - mirror streaming engine
            primary_res = err
            logger.warning("Primary ASR failed: %s", err)
    if secondary_task is not None:
        try:
            secondary_res = await secondary_task
        except Exception as err:  # noqa: BLE001
            secondary_res = err
            logger.warning("Secondary ASR failed: %s", err)

    primary_result = (
        None if isinstance(primary_res, Exception) else primary_res
    )
    secondary_result = (
        None if isinstance(secondary_res, Exception) else secondary_res
    )
    if primary_result is None and secondary_result is None:
        raise HTTPException(
            status_code=502, detail="all configured ASR models failed"
        )

    detected_lang = language or ""
    if primary_result and not secondary_result:
        text = str(primary_result.get("transcription") or "").strip()
        detected_lang = (
            primary_result.get("detected_language") or detected_lang
        )
    elif secondary_result and not primary_result:
        text = str(secondary_result.get("transcription") or "").strip()
    else:
        fused = choose_fused_result(
            primary_result,
            secondary_result,
            hotwords=hw_list,
            similarity_threshold=cfg.fusion_similarity_threshold,
            min_primary_score=cfg.fusion_min_primary_score,
            max_repetition_ratio=cfg.fusion_max_repetition_ratio,
            disagreement_threshold=cfg.fusion_disagreement_threshold,
            hotword_boost=cfg.fusion_hotword_boost,
            primary_score_margin=cfg.fusion_primary_score_margin,
        )
        text = str(fused.get("text") or "").strip()
        if primary_result and primary_result.get("detected_language"):
            detected_lang = primary_result["detected_language"]

    return {
        "type": "final",
        "text": text,
        "language": detected_lang,
        "duration_sec": round(duration_sec, 3),
    }


@app.post("/api/emotion/upload")
async def emotion_upload(
    audio: UploadFile = File(...),
    mode: str = Form(""),
    language: str = Form(""),
):
    """One-shot emotion inference over an uploaded clip."""
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    cap = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))
    wav_bytes, duration_sec = _wav_to_pcm_capped(raw, cap)
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    chosen_mode = normalize_mode(mode or getattr(cfg, "emotion_task_mode", "ser"))

    try:
        result = await query_emotion_model(
            wav_b64,
            mode=chosen_mode,
            base_url=cfg.emotion_vllm_base_url,
            model_name=cfg.emotion_vllm_model_name,
            timeout=cfg.emotion_request_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Emotion upload inference failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    payload: dict = {
        "type": "final_emotion",
        "mode": chosen_mode,
        "label": result.get("label", ""),
        "text": result.get("text", ""),
        "duration_sec": round(duration_sec, 3),
    }
    raw_text = result.get("raw_text", "")
    if raw_text and raw_text != payload["text"]:
        payload["raw_text"] = raw_text
    if language:
        payload["language"] = language
    return payload


@app.post("/api/tsasr/upload")
async def tsasr_upload(
    audio: UploadFile = File(...),
    enrollment_wav_base64: str = Form(...),
    language: str = Form(""),
    hotwords: str = Form(""),
    voice_traits: str = Form(""),
):
    """One-shot TS-ASR over an uploaded mixed clip + enrollment WAV.

    Enrollment is delivered as a base64-encoded WAV form field instead of a
    second multipart file because the frontend already keeps it in that
    shape (``enrollWavB64``) for the live-mic flow; reusing the same field
    avoids redundant decode/encode work in the browser.
    """
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    cap = float(getattr(cfg, "tsasr_max_audio_seconds", 0.0))
    wav_bytes, duration_sec = _wav_to_pcm_capped(raw, cap)
    mixed_b64 = base64.b64encode(wav_bytes).decode("ascii")

    try:
        enrollment = decode_enrollment(
            enrollment_wav_base64,
            min_sec=float(getattr(cfg, "tsasr_enrollment_min_sec", 1.0)),
            max_sec=float(getattr(cfg, "tsasr_enrollment_max_sec", 5.0)),
            audio_format="wav",
        )
    except EnrollmentError as err:
        raise HTTPException(
            status_code=400,
            detail={"code": f"enrollment_{err.code}", "message": str(err)},
        ) from err

    base_url = (
        getattr(cfg, "tsasr_base_url", "") or cfg.vllm_base_url
    )
    model_name = (
        getattr(cfg, "tsasr_model_name", "") or cfg.vllm_model_name
    )
    timeout = float(
        getattr(cfg, "tsasr_request_timeout", 0)
        or cfg.asr_request_timeout
    )
    hw_list = (
        _parse_csv(hotwords)
        if bool(getattr(cfg, "tsasr_enable_hotwords", False))
        else None
    )
    traits = voice_traits.strip() or None

    try:
        result = await query_tsasr_model(
            mixed_b64,
            enrollment.wav_base64,
            hotwords=hw_list,
            voice_traits=traits,
            base_url=base_url,
            model_name=model_name,
            timeout=timeout,
            enrollment_duration_sec=enrollment.duration_sec,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("TS-ASR upload inference failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = str(result.get("transcription") or "").strip()
    detected_lang = result.get("detected_language") or language or ""
    return {
        "type": "final",
        "task": "tsasr",
        "text": text,
        "language": detected_lang,
        "duration_sec": round(duration_sec, 3),
        # Echo the (possibly trimmed) mixed audio back so the client can wire
        # up a replay button just like the streaming ``final`` payload does.
        "audio_b64": mixed_b64,
    }


# Static mount comes last so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
