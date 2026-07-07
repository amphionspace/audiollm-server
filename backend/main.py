import asyncio
import base64
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import websockets
from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.params import Form as FormParam
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from .asr.enrollment import (
    EnrollmentError,
    decode_and_validate_audio_bytes,
    get_enrollment_store,
)
from .asr.jobs import get_transcription_job_store
from .asr.k2.client import close_k2_channels
from .asr.oneshot import OneshotAsrError, run_oneshot_asr
from .asr.recall import (
    add_hotwords as add_recall_hotwords,
)
from .asr.recall import (
    clear_hotword_pool,
    list_hotword_pool,
    reload_hotword_pool,
)
from .asr.recall import (
    delete_enrollment as delete_triton_enrollment,
)
from .asr.recall import (
    delete_hotwords as delete_recall_hotwords,
)
from .asr.recall import (
    get_enrollment as get_triton_enrollment,
)
from .asr.recall import (
    upsert_enrollment as upsert_triton_enrollment,
)
from .asr.transcribe import float_pcm_to_i16_bytes
from .audio.utils import (
    mp3_bytes_to_pcm_16k_mono,
    pcm_to_wav_base64,
    wav_base64_to_pcm_16k_mono,
    wav_bytes_to_pcm_16k_mono,
)
from .config import SAMPLE_RATE, load_config, load_transcribe_config
from .emotion.client import query_emotion_model
from .emotion.jobs import JobQueueFullError, get_emotion_job_store
from .emotion.service import EmotionDecodeError, decode_wav_capped
from .emotion_spec.jobs import get_emotion_spec_job_store
from .http_client import close_client
from .recall_user import HotwordPoolIdError, normalize_hotword_pool_id
from .streaming import (
    AstV3Protocol,
    K2SegmentedStream,
    StreamingSession,
    VadSegmentedStream,
)
from .tasks import AsrTaskEngine, EmotionTaskEngine
from .text_cleanup import clean_asr_text
from .text_cleanup.client import TextCleanupConfigError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_k2_channels()
    await close_client()


app = FastAPI(title="AudioLLM Server", lifespan=lifespan)


@app.websocket("/transcribe-streaming")
async def transcribe_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info("Transcribe-streaming connected (language=%s)", language)
    cfg = load_config()
    session = StreamingSession(
        websocket,
        stream=K2SegmentedStream() if cfg.k2_enabled else VadSegmentedStream(),
        engine=AsrTaskEngine(),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/tuling/ast/v3")
async def tuling_ast_v3_ws(websocket: WebSocket):
    """iFlytek Tuling AST v3 streaming ASR.

    Same VAD-segmented dual-ASR pipeline as ``/transcribe-streaming``, but the
    on-the-wire framing is the AST v3 ``header/parameter/payload`` envelope:
    audio arrives base64-encoded inside JSON frames, ``header.status`` (0/1/2)
    drives start/stop, and results are repackaged into the ``payload.result``
    lattice. ``AstV3Protocol`` owns all of that translation; the session,
    stream, and engine are the shared ones. ``emit_timing`` lets the engine
    surface segment ``bg``/``ed`` to the protocol.

    See ``docs/protocols/tuling-ast-v3-protocol.md``.
    """
    await websocket.accept()
    logger.info("Tuling AST v3 connected (/tuling/ast/v3)")
    # Endpoint policy: primary-only (no secondary / no local Qwen / no fusion),
    # with the primary pinned to the AST v3-specific upstream when configured
    # (empty astv3_vllm_* falls back to the global primary). These are forced
    # overrides (see StreamingSession._config_overrides), re-applied after the
    # client's start.config so a client cannot re-enable secondary via
    # parameter.asr_config.
    cfg = load_config()
    astv3_overrides: dict[str, object] = {"enable_secondary_asr": False}
    if cfg.astv3_vllm_base_url:
        astv3_overrides["vllm_base_url"] = cfg.astv3_vllm_base_url
    if cfg.astv3_vllm_model_name:
        astv3_overrides["vllm_model_name"] = cfg.astv3_vllm_model_name
    if cfg.astv3_vllm_prompt_template:
        astv3_overrides["vllm_prompt_template"] = cfg.astv3_vllm_prompt_template
    session = StreamingSession(
        websocket,
        stream=K2SegmentedStream() if cfg.k2_enabled else VadSegmentedStream(),
        engine=AsrTaskEngine(emit_timing=True),
        protocol=AstV3Protocol(),
        config_overrides=astv3_overrides,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


# Hard-coded remote AST v3 backend that the "实时语音识别（测试用）" page targets.
# It speaks plaintext ws://, so an HTTPS-served frontend (playground.amphion.top)
# cannot open it directly — the browser's mixed-content policy forbids ws:// from
# an https:// page. The same-origin proxy below bridges the browser's wss://
# connection to it. Temporary test scaffolding: the address is intentionally
# pinned here, not exposed as config.
ASTV3_TEST_PROXY_UPSTREAM = "ws://159.138.9.106:18082/tuling/ast/v3"


async def _astv3_proxy_pump_to_upstream(client: WebSocket, upstream) -> None:
    """Relay browser -> upstream frames verbatim (text or binary)."""
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is not None:
                await upstream.send(text)
                continue
            data = message.get("bytes")
            if data is not None:
                await upstream.send(data)
    except (WebSocketDisconnect, websockets.ConnectionClosed, RuntimeError):
        pass
    finally:
        # Closing the upstream unblocks the opposite pump's async-for so the
        # whole proxy tears down once either side goes away.
        await upstream.close()


async def _astv3_proxy_pump_to_client(upstream, client: WebSocket) -> None:
    """Relay upstream -> browser frames verbatim (text or binary)."""
    try:
        async for message in upstream:
            if isinstance(message, (bytes, bytearray)):
                await client.send_bytes(bytes(message))
            else:
                await client.send_text(message)
    except (websockets.ConnectionClosed, WebSocketDisconnect, RuntimeError):
        pass
    finally:
        try:
            await client.close()
        except RuntimeError:
            # client transport already closed; nothing to do
            pass


@app.websocket("/astv3-test-proxy")
async def astv3_test_proxy_ws(websocket: WebSocket):
    """Same-origin WebSocket proxy for the AST v3 test page.

    The "实时语音识别（测试用）" page is served over HTTPS but its target backend
    speaks plaintext ws:// (``ASTV3_TEST_PROXY_UPSTREAM``), which the browser's
    mixed-content policy forbids opening from an HTTPS page. This endpoint accepts
    the browser's same-origin (wss://) connection and relays every frame, in both
    directions, to/from that upstream without inspecting the AST v3 envelope. It
    is a transparent byte pump, so the on-the-wire contract is identical to
    ``/tuling/ast/v3`` (see ``docs/protocols/tuling-ast-v3-protocol.md``).

    Temporary test scaffolding: the upstream address is hard-coded.
    """
    await websocket.accept()
    logger.info("AST v3 test proxy connected -> %s", ASTV3_TEST_PROXY_UPSTREAM)
    try:
        async with websockets.connect(
            ASTV3_TEST_PROXY_UPSTREAM, max_size=None, open_timeout=10
        ) as upstream:
            await asyncio.gather(
                _astv3_proxy_pump_to_upstream(websocket, upstream),
                _astv3_proxy_pump_to_client(upstream, websocket),
            )
    except WebSocketDisconnect:
        # Browser hung up before/while the upstream was being dialed.
        pass
    except Exception as exc:  # upstream connect / handshake failure
        logger.warning("AST v3 test proxy upstream error: %s", exc)
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass


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
# (emotion 20s tail, ASR 60s tail) are still applied here so a malicious or
# buggy client can't bypass them by switching from WS to REST.

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


async def _read_audio_bytes(
    audio: UploadFile, max_bytes: int = _MAX_UPLOAD_BYTES
) -> bytes:
    """Read a multipart audio upload, enforcing the byte cap.

    UploadFile.read with no argument loads into memory; the size check is
    primarily a guard against accidental huge uploads, not a streaming
    safeguard (we need the full payload for vLLM anyway). Long-audio
    transcription passes its own (much larger) cap.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio file is empty")
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"audio file exceeds {max_bytes} bytes",
        )
    return raw


def _upload_audio_to_pcm_capped(
    raw: bytes,
    audio: UploadFile,
    max_seconds: float,
) -> tuple[bytes, np.ndarray, float]:
    """Decode an uploaded ASR clip to 16 kHz mono and tail-trim it.

    Returns (re_encoded_wav_bytes, pcm_16k_mono, duration_sec). When no trim is
    needed the re-encoded WAV is byte-equivalent to ``pcm_to_wav_base64(pcm)``.
    """
    try:
        audio_format = _infer_enrollment_audio_format(audio, raw)
        if audio_format in {"wav", "wave"}:
            pcm = wav_bytes_to_pcm_16k_mono(raw)
        elif audio_format == "mp3":
            pcm = mp3_bytes_to_pcm_16k_mono(raw)
        else:
            raise ValueError("unsupported audio format; supported: wav, mp3")
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc
    return _pcm_to_capped_wav(pcm, max_seconds)


def _wav_to_pcm_capped(raw: bytes, max_seconds: float) -> tuple[bytes, np.ndarray, float]:
    """Decode a WAV blob to 16 kHz mono and tail-trim to ``max_seconds``."""
    try:
        pcm = wav_bytes_to_pcm_16k_mono(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc
    return _pcm_to_capped_wav(pcm, max_seconds)


def _pcm_to_capped_wav(
    pcm: np.ndarray,
    max_seconds: float,
) -> tuple[bytes, np.ndarray, float]:
    """Validate, tail-trim, and encode 16 kHz mono float PCM as WAV bytes."""
    import io
    import wave

    if pcm.size == 0:
        raise HTTPException(status_code=400, detail="audio decoded to empty PCM")
    duration = pcm.size / SAMPLE_RATE
    if max_seconds > 0 and duration > max_seconds:
        # Match streaming engines: keep the trailing window. Emotion picks
        # the tail because the most recent emotion is what users care about;
        # we use the same convention for ASR for consistency.
        keep = int(SAMPLE_RATE * max_seconds)
        pcm = pcm[-keep:]
        duration = pcm.size / SAMPLE_RATE
    new_b64 = pcm_to_wav_base64(pcm.astype(np.float32, copy=False))
    new_bytes = base64.b64decode(new_b64)
    with wave.open(io.BytesIO(new_bytes), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
    return new_bytes, pcm.astype(np.float32, copy=False), duration


def _infer_enrollment_audio_format(audio: UploadFile, raw: bytes) -> str:
    """Resolve the supported enrollment upload format.

    PCM has no container metadata, so callers must identify it by file suffix
    or content type. WAV/MP3 can also be detected by their common magic bytes.
    """
    filename = (audio.filename or "").lower()
    content_type = (audio.content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix in {"wav", "wave", "mp3", "pcm", "raw", "s16le"}:
        return "wav" if suffix == "wave" else suffix
    if content_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return "wav"
    if content_type in {"audio/mpeg", "audio/mp3"}:
        return "mp3"
    if content_type in {
        "audio/pcm",
        "audio/l16",
        "audio/x-raw",
        "application/octet-stream+pcm",
    }:
        return "pcm"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
        return "wav"
    if raw.startswith(b"ID3") or (
        len(raw) >= 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0
    ):
        return "mp3"
    return ""


def _emotion_result_payload(
    result: object,
    *,
    mode: str,
    duration_sec: float,
    language: str,
) -> dict:
    if isinstance(result, Exception):
        return {
            "type": "error",
            "mode": mode,
            "message": str(result),
            "error_type": result.__class__.__name__,
        }
    payload = {
        "type": "final_emotion",
        "mode": mode,
        "label": result.get("label", ""),
        "text": result.get("text", ""),
        "duration_sec": round(duration_sec, 3),
    }
    if language:
        payload["language"] = language
    return payload


def _public_asr_payload(result: dict) -> dict:
    return {
        "text": result.get("text", ""),
        "language": result.get("language", ""),
    }


def _public_cleanup_payload(result: dict) -> dict:
    return {
        "text": result.get("text", ""),
    }


def _resolve_enrollment_b64(enrollment_id: str | None) -> str | None:
    """Look up an enrollment id, refreshing its TTL. Missing/expired ids
    return ``None`` (caller decides to error or silently fall back)."""
    if not enrollment_id:
        return None
    entry = get_enrollment_store().get(enrollment_id)
    return entry.wav_base64 if entry is not None else None


def _enrollment_store_owner_id(cfg) -> str:
    return normalize_hotword_pool_id(None, default=cfg.hotword_pool_id)


async def _resolve_enrollment_for_request(
    cfg,
    enrollment_id: str | None,
) -> tuple[str | None, str | None]:
    """Return ``(legacy_b64, triton_id)`` for the configured enrollment store."""
    if not enrollment_id:
        return None, None
    if not cfg.enable_triton_enrollment_store:
        return _resolve_enrollment_b64(enrollment_id), None
    ident = str(enrollment_id or "").strip()
    if not ident:
        return None, None
    try:
        summary = await get_triton_enrollment(
            enrollment_id=ident,
            enrollment_scope_id=_enrollment_store_owner_id(cfg),
        )
    except Exception as exc:
        logger.warning("Triton enrollment lookup failed; falling back to plain ASR: %s", exc)
        return None, None
    if str(summary.get("status", "")).lower() == "ok":
        return None, ident
    logger.warning("Triton enrollment id %s not found / expired", ident)
    return None, None


def _public_enrollment_status(
    enrollment_id: str,
    summary: dict[str, object] | None,
) -> dict[str, object]:
    if not summary:
        return {
            "enrollment_id": enrollment_id,
            "available": False,
            "reason": "not_found",
        }
    available_raw = summary.get("available")
    if isinstance(available_raw, bool):
        available = available_raw
    else:
        status = str(summary.get("status", "") or "").strip().lower()
        available = status in {"ok", "available", "found"}
    reason = str(summary.get("reason") or summary.get("code") or "").strip()
    if not reason:
        reason = "ok" if available else "not_found"
    return {
        "enrollment_id": enrollment_id,
        "available": available,
        "reason": reason,
    }


async def _run_dual_asr_upload(
    wav_b64: str,
    *,
    cfg,
    hotwords: list[str],
    language: str,
    audio_pcm: np.ndarray,
    enrollment_b64: str | None = None,
    enrollment_id: str | None = None,
    enrollment_scope_id: str | None = None,
    hotword_pool_id: str | None = None,
) -> dict:
    """Route-facing wrapper over :func:`run_oneshot_asr`.

    Translates the framework-free :class:`OneshotAsrError` into the 502 the
    REST contract documents for "every configured model failed".
    """
    try:
        return await run_oneshot_asr(
            wav_b64,
            cfg=cfg,
            hotwords=hotwords,
            language=language,
            audio_pcm=audio_pcm,
            enrollment_b64=enrollment_b64,
            enrollment_id=enrollment_id,
            enrollment_scope_id=enrollment_scope_id,
            hotword_pool_id=hotword_pool_id,
        )
    except OneshotAsrError as exc:
        raise HTTPException(status_code=502, detail=exc.to_detail()) from exc


@app.post("/api/asr/enrollment")
async def asr_enrollment_create(audio: UploadFile = File(...)):
    """Cache a target-speaker enrollment clip and return its opaque id.

    The frontend uploads a 1–8 s clip once (file or mic recording) and
    then passes the returned ``enrollment_id`` on every ``/api/asr/upload``
    call and the WS ``start`` payload. The server validates duration,
    canonicalises to 16 kHz mono WAV, and stores the base64 result so
    primary inference can splice it into the dual-audio prompt without
    re-decoding on every segment.
    """
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    try:
        canonical_b64, duration_sec = decode_and_validate_audio_bytes(
            raw,
            audio_format=_infer_enrollment_audio_format(audio, raw),
            min_sec=cfg.asr_enrollment_min_sec,
            max_sec=cfg.asr_enrollment_max_sec,
        )
    except EnrollmentError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    if cfg.enable_triton_enrollment_store:
        enrollment_id = secrets.token_urlsafe(16)
        pcm = wav_base64_to_pcm_16k_mono(canonical_b64)
        try:
            await upsert_triton_enrollment(
                pcm,
                enrollment_id=enrollment_id,
                enrollment_scope_id=_enrollment_store_owner_id(cfg),
                sample_rate=SAMPLE_RATE,
            )
        except Exception as exc:
            logger.exception("Triton enrollment upsert failed")
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "triton_enrollment_failed",
                    "message": str(exc),
                },
            ) from exc
        return {
            "enrollment_id": enrollment_id,
            "duration_sec": round(duration_sec, 3),
        }

    store = get_enrollment_store()
    store.configure(
        ttl_sec=cfg.asr_enrollment_ttl_sec,
        max_entries=cfg.asr_enrollment_max_entries,
    )
    entry = store.put(canonical_b64, duration_sec)
    return {
        "enrollment_id": entry.enrollment_id,
        "duration_sec": round(duration_sec, 3),
    }


@app.delete("/api/asr/enrollment/{enrollment_id}")
async def asr_enrollment_delete(enrollment_id: str):
    """Drop a previously uploaded enrollment clip.

    Returning 204 on missing ids keeps the frontend's "clear" button
    idempotent — repeated clears never error out.
    """
    cfg = load_config()
    if cfg.enable_triton_enrollment_store:
        try:
            await delete_triton_enrollment(
                enrollment_id=enrollment_id,
                enrollment_scope_id=_enrollment_store_owner_id(cfg),
            )
        except Exception:
            logger.warning("Triton enrollment delete failed", exc_info=True)
    else:
        get_enrollment_store().delete(enrollment_id)
    return Response(status_code=204)


@app.get("/api/asr/enrollment/{enrollment_id}")
async def asr_enrollment_get(enrollment_id: str):
    """Return a public availability summary for an enrollment id.

    This is a diagnostic/status API only: it never exposes raw enrollment
    audio, PCM, embedding tensors, or private metadata.
    """
    cfg = load_config()
    ident = str(enrollment_id or "").strip()
    if cfg.enable_triton_enrollment_store:
        try:
            summary = await get_triton_enrollment(
                enrollment_id=ident,
                enrollment_scope_id=_enrollment_store_owner_id(cfg),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Triton enrollment status lookup failed: %s", exc)
            return {
                "enrollment_id": ident,
                "available": False,
                "reason": "upstream_unavailable",
            }
        return _public_enrollment_status(ident, summary)

    entry = get_enrollment_store().get(ident)
    return {
        "enrollment_id": ident,
        "available": entry is not None,
        "reason": "ok" if entry is not None else "not_found",
    }


def _hotword_pool_payload(body: dict) -> list[str]:
    values = body.get("hotwords", body.get("words", []))
    if not isinstance(values, list):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_hotwords",
                "message": "hotwords must be a JSON array of strings",
            },
        )
    words = [str(item).strip() for item in values if str(item).strip()]
    if not words:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "empty_hotwords",
                "message": "hotwords must contain at least one non-empty string",
            },
        )
    return words


def _resolve_hotword_pool_id(raw_hotword_pool_id: object | None, cfg) -> str:
    try:
        return normalize_hotword_pool_id(
            raw_hotword_pool_id,
            default=cfg.hotword_pool_id,
        )
    except HotwordPoolIdError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_hotword_pool_id",
                "message": str(exc),
            },
        ) from exc


def _resolve_hotword_pool_id_from_body_or_query(
    body: dict | None,
    query_hotword_pool_id: str,
    cfg,
) -> str:
    raw_body_hotword_pool_id = None
    if body is not None and "hotword_pool_id" in body:
        raw_body_hotword_pool_id = body.get("hotword_pool_id")
    query_value = str(query_hotword_pool_id or "").strip()
    body_value = (
        str(raw_body_hotword_pool_id or "").strip()
        if raw_body_hotword_pool_id is not None
        else ""
    )
    if query_value and body_value:
        resolved_query = _resolve_hotword_pool_id(query_value, cfg)
        resolved_body = _resolve_hotword_pool_id(body_value, cfg)
        if resolved_query != resolved_body:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_hotword_pool_id",
                    "message": "query and body hotword_pool_id must match",
                },
            )
        return resolved_query
    return _resolve_hotword_pool_id(body_value or query_value, cfg)


def _reject_legacy_hotword_user_id(value: object) -> None:
    if isinstance(value, FormParam):
        return
    if value is not None:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_hotword_pool_id",
                "message": "user_id is no longer supported; use hotword_pool_id",
            },
        )


def _recall_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail={
            "code": "triton_recall_unavailable",
            "message": "Triton hotword recall service request failed",
            "detail": str(exc),
        },
    )


def _count_or_len(payload: dict, count_key: str, list_key: str) -> int:
    value = payload.get(count_key)
    if isinstance(value, int):
        return value
    items = payload.get(list_key)
    if isinstance(items, list):
        return len(items)
    return 0


def _hotword_review_response(payload: dict) -> dict:
    """Project upstream hotword results onto the TD-Tech review contract."""
    if not isinstance(payload, dict):
        return payload
    action = str(payload.get("action") or "").lower()
    common = {
        "action": payload.get("action") or action,
        "status": payload.get("status", "ok"),
        "hotword_pool_id": payload.get("hotword_pool_id"),
        "total_count": payload.get("total_count", 0),
    }
    if payload.get("message") is not None:
        common["message"] = payload.get("message")
    if action == "add" or "added" in payload or "skipped_duplicates" in payload:
        ignored = payload.get("ignored_hotwords")
        if not isinstance(ignored, list):
            ignored = payload.get("invalid")
        if not isinstance(ignored, list):
            ignored = []
        return {
            **common,
            "action": "add",
            "hotwords": payload.get("hotwords", []),
            "added_count": _count_or_len(payload, "added", "hotwords"),
            "duplicate_count": _count_or_len(payload, "skipped_duplicates", "duplicates"),
            "invalid_count": _count_or_len(payload, "invalid_count", "invalid"),
            "ignored_hotwords": ignored,
        }
    if action == "delete" or "deleted" in payload or "missing" in payload:
        missing = payload.get("missing_hotwords")
        if not isinstance(missing, list):
            missing = payload.get("missing")
        if not isinstance(missing, list):
            missing = []
        return {
            **common,
            "action": "delete",
            "hotwords": payload.get("hotwords", []),
            "deleted_count": _count_or_len(payload, "deleted", "hotwords"),
            "missing_count": _count_or_len(payload, "missing_count", "missing"),
            "missing_hotwords": missing,
        }
    return payload


@app.get("/api/asr/hotword-pool")
async def asr_hotword_pool_list(
    query: str = "",
    limit: int = 50,
    offset: int = 0,
    hotword_pool_id: str = "",
    user_id: str | None = None,
):
    _reject_legacy_hotword_user_id(user_id)
    cfg = load_config()
    resolved_hotword_pool_id = _resolve_hotword_pool_id(
        hotword_pool_id,
        cfg,
    )
    try:
        return await list_hotword_pool(
            query=query or None,
            limit=max(0, min(int(limit), 1000)),
            offset=max(0, int(offset)),
            hotword_pool_id=resolved_hotword_pool_id,
        )
    except Exception as exc:  # noqa: BLE001 - route maps upstream failures
        raise _recall_error(exc) from exc


@app.post("/api/asr/hotword-pool")
async def asr_hotword_pool_add(body: dict = Body(...)):
    _reject_legacy_hotword_user_id(
        body.get("user_id") if "user_id" in body else None
    )
    cfg = load_config()
    resolved_hotword_pool_id = _resolve_hotword_pool_id(
        body.get("hotword_pool_id"),
        cfg,
    )
    try:
        return _hotword_review_response(
            await add_recall_hotwords(
                _hotword_pool_payload(body),
                hotword_pool_id=resolved_hotword_pool_id,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _recall_error(exc) from exc


@app.delete("/api/asr/hotword-pool")
@app.post("/api/asr/hotword-pool/delete")
async def asr_hotword_pool_delete(body: dict = Body(...)):
    _reject_legacy_hotword_user_id(
        body.get("user_id") if "user_id" in body else None
    )
    cfg = load_config()
    resolved_hotword_pool_id = _resolve_hotword_pool_id(
        body.get("hotword_pool_id"),
        cfg,
    )
    try:
        return _hotword_review_response(
            await delete_recall_hotwords(
                _hotword_pool_payload(body),
                hotword_pool_id=resolved_hotword_pool_id,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _recall_error(exc) from exc


@app.post("/api/asr/hotword-pool/clear")
async def asr_hotword_pool_clear(
    hotword_pool_id: str = "",
    user_id: str | None = None,
    body: dict | None = Body(default=None),
):
    body = body if isinstance(body, dict) else None
    _reject_legacy_hotword_user_id(user_id)
    _reject_legacy_hotword_user_id(
        body.get("user_id") if body is not None and "user_id" in body else None
    )
    cfg = load_config()
    resolved_hotword_pool_id = _resolve_hotword_pool_id_from_body_or_query(
        body,
        hotword_pool_id,
        cfg,
    )
    try:
        return await clear_hotword_pool(hotword_pool_id=resolved_hotword_pool_id)
    except Exception as exc:  # noqa: BLE001
        raise _recall_error(exc) from exc


@app.post("/api/asr/hotword-pool/reload")
async def asr_hotword_pool_reload(
    hotword_pool_id: str = "",
    user_id: str | None = None,
    body: dict | None = Body(default=None),
):
    body = body if isinstance(body, dict) else None
    _reject_legacy_hotword_user_id(user_id)
    _reject_legacy_hotword_user_id(
        body.get("user_id") if body is not None and "user_id" in body else None
    )
    cfg = load_config()
    resolved_hotword_pool_id = _resolve_hotword_pool_id_from_body_or_query(
        body,
        hotword_pool_id,
        cfg,
    )
    try:
        return await reload_hotword_pool(hotword_pool_id=resolved_hotword_pool_id)
    except Exception as exc:  # noqa: BLE001
        raise _recall_error(exc) from exc


@app.post("/api/asr/upload")
async def asr_upload(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
    enrollment_id: str = Form(""),
    hotword_pool_id: str = Form(""),
    user_id: str | None = Form(None),
):
    """One-shot ASR over an uploaded clip.

    Mirrors :class:`AsrTaskEngine.handle_segment` but operates on the entire
    clip in a single dual-model call (no VAD segmentation, no partials).
    Returns the same fields the streaming ``final`` event carries.

    When ``enrollment_id`` resolves to a cached enrollment clip the primary
    model is prompted with the dual-audio TS-ASR template (task 5/6 of the
    v4 prompt spec). Unknown / expired ids fall back to plain ASR so a
    stale id from a long-running tab does not break uploads.
    """
    raw = await _read_audio_bytes(audio)
    wav_bytes, audio_pcm, duration_sec = _upload_audio_to_pcm_capped(
        raw,
        audio,
        _ASR_MAX_SECONDS,
    )
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    cfg = load_config()
    _reject_legacy_hotword_user_id(user_id)
    resolved_hotword_pool_id = _resolve_hotword_pool_id(
        hotword_pool_id,
        cfg,
    )
    hw_list = _parse_csv(hotwords)
    enrollment_b64, triton_enrollment_id = await _resolve_enrollment_for_request(
        cfg,
        enrollment_id,
    )
    asr_result = await _run_dual_asr_upload(
        wav_b64,
        cfg=cfg,
        hotwords=hw_list,
        language=language,
        audio_pcm=audio_pcm,
        enrollment_b64=enrollment_b64,
        enrollment_id=triton_enrollment_id,
        enrollment_scope_id=_enrollment_store_owner_id(cfg)
        if triton_enrollment_id
        else None,
        hotword_pool_id=resolved_hotword_pool_id,
    )

    return {
        "type": "final",
        "text": asr_result["text"],
        "language": asr_result["language"],
        "duration_sec": round(duration_sec, 3),
        "effective_hotwords": list(asr_result.get("effective_hotwords") or []),
        "enrollment_used": enrollment_b64 is not None or triton_enrollment_id is not None,
    }


@app.post("/api/asr/transcriptions", status_code=202)
async def asr_transcription_create(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
    hotword_pool_id: str = Form(""),
    user_id: str | None = Form(None),
):
    """Enqueue offline transcription of a long recording (meeting minutes).

    Unlike ``/api/asr/upload`` (single clip, 60 s tail-trim, synchronous),
    this accepts whole recordings up to ``transcribe_max_audio_sec``: the
    audio is VAD-segmented server-side with the same parameters as the
    streaming endpoints and each segment is transcribed via the shared
    one-shot dual-ASR path. Recordings longer than the cap are REJECTED
    (400) rather than trimmed — silently losing meeting content is worse
    than asking the client to split the file.

    Poll ``GET /api/asr/transcriptions/{job_id}`` for progress and the final
    segment list (``start_ms``/``end_ms`` per segment + ``full_text``).

    No ``enrollment_id`` here on purpose: target-speaker filtering keeps a
    single speaker and drops everyone else, which is the opposite of what a
    multi-speaker meeting transcript needs.
    """
    # Transcription-specific view: rest.routes.transcribe bindings (model
    # choice, fusion switch) layered over the shared REST defaults.
    cfg = load_transcribe_config()
    _reject_legacy_hotword_user_id(user_id)
    resolved_hotword_pool_id = _resolve_hotword_pool_id(
        hotword_pool_id,
        cfg,
    )
    raw = await _read_audio_bytes(audio, max_bytes=cfg.transcribe_max_upload_bytes)
    try:
        pcm = wav_bytes_to_pcm_16k_mono(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc
    del raw
    if pcm.size == 0:
        raise HTTPException(status_code=400, detail="audio decoded to empty PCM")

    duration_sec = pcm.size / SAMPLE_RATE
    max_sec = float(cfg.transcribe_max_audio_sec)
    if max_sec > 0 and duration_sec > max_sec:
        raise HTTPException(
            status_code=400,
            detail=(
                f"audio duration {duration_sec:.1f}s exceeds the "
                f"{max_sec:.0f}s transcription cap; split the recording"
            ),
        )

    pcm_i16 = float_pcm_to_i16_bytes(pcm)
    del pcm

    store = get_transcription_job_store()
    store.configure(cfg)
    try:
        job = await store.submit(
            pcm_i16,
            duration_sec=duration_sec,
            language=language,
            hotwords=_parse_csv(hotwords),
            hotword_pool_id=resolved_hotword_pool_id,
            cfg=cfg,
        )
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status,
            "poll_url": f"/api/asr/transcriptions/{job.job_id}",
            "duration_sec": round(duration_sec, 3),
        },
    )


@app.get("/api/asr/transcriptions/{job_id}")
async def asr_transcription_get(job_id: str):
    """Poll offline transcription job status, progress, and result."""
    store = get_transcription_job_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_poll_dict()


@app.post("/api/emotion/jobs", status_code=202)
async def emotion_create_job(
    audio: UploadFile = File(...),
    mode: str = Form(""),
    language: str = Form(""),
):
    """Enqueue whole-utterance emotion inference; poll GET /api/emotion/jobs/{id}."""
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    cap = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))
    try:
        decode_wav_capped(raw, cap)
    except EmotionDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_emotion_job_store()
    store.configure(cfg)
    try:
        job = await store.submit(
            raw,
            mode=mode,
            language=language,
            cfg=cfg,
        )
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc

    poll_url = f"/api/emotion/jobs/{job.job_id}"
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status,
            "poll_url": poll_url,
        },
    )


@app.get("/api/emotion/jobs/{job_id}")
async def emotion_get_job(job_id: str):
    """Poll async emotion job status and result."""
    store = get_emotion_job_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_poll_dict()


@app.post("/api/emotion-spec/jobs", status_code=202)
async def emotion_spec_create_job(
    audio: UploadFile = File(...),
    mode: str = Form(""),
    language: str = Form(""),
):
    """Enqueue whole-utterance AmphionSPEC inference; poll GET /api/emotion-spec/jobs/{id}.

    Independent of ``/api/emotion/jobs`` — separate queue, separate
    concurrency budget, separate vLLM endpoint (cfg.emotion_spec_vllm_*).
    ``mode`` accepts ``ser`` or ``sepc`` (alias ``spec`` is normalized to
    ``sepc``); empty falls back to ``cfg.emotion_spec_task_mode``.
    """
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    cap = float(getattr(cfg, "emotion_spec_max_audio_seconds", 0.0))
    try:
        decode_wav_capped(raw, cap)
    except EmotionDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_emotion_spec_job_store()
    store.configure(cfg)
    try:
        job = await store.submit(
            raw,
            mode=mode,
            language=language,
            cfg=cfg,
        )
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc

    poll_url = f"/api/emotion-spec/jobs/{job.job_id}"
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status,
            "poll_url": poll_url,
        },
    )


@app.get("/api/emotion-spec/jobs/{job_id}")
async def emotion_spec_get_job(job_id: str):
    """Poll async AmphionSPEC job status and result."""
    store = get_emotion_spec_job_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_poll_dict()


@app.post("/api/audio/analyze")
async def audio_analyze(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
    emotion_mode: str = Form("both"),
    enrollment_id: str = Form(""),
    hotword_pool_id: str = Form(""),
    user_id: str | None = Form(None),
):
    """One-shot audio analysis: ASR raw output + cleaned text + emotion."""
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    _reject_legacy_hotword_user_id(user_id)
    resolved_hotword_pool_id = _resolve_hotword_pool_id(
        hotword_pool_id,
        cfg,
    )
    hw_list = _parse_csv(hotwords)
    enrollment_b64, triton_enrollment_id = await _resolve_enrollment_for_request(
        cfg,
        enrollment_id,
    )

    asr_wav_bytes, asr_pcm, duration_sec = _wav_to_pcm_capped(raw, _ASR_MAX_SECONDS)
    asr_wav_b64 = base64.b64encode(asr_wav_bytes).decode("ascii")

    emotion_cap = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))
    emotion_wav_bytes, _, emotion_duration_sec = _wav_to_pcm_capped(raw, emotion_cap)
    emotion_wav_b64 = base64.b64encode(emotion_wav_bytes).decode("ascii")

    asr_task = asyncio.create_task(
        _run_dual_asr_upload(
            asr_wav_b64,
            cfg=cfg,
            hotwords=hw_list,
            language=language,
            audio_pcm=asr_pcm,
            enrollment_b64=enrollment_b64,
            enrollment_id=triton_enrollment_id,
            enrollment_scope_id=_enrollment_store_owner_id(cfg)
            if triton_enrollment_id
            else None,
            hotword_pool_id=resolved_hotword_pool_id,
        )
    )
    emotion_ser_task = asyncio.create_task(
        query_emotion_model(
            emotion_wav_b64,
            mode="ser",
            base_url=cfg.emotion_vllm_base_url,
            model_name=cfg.emotion_vllm_model_name,
            timeout=cfg.emotion_request_timeout,
        )
    )
    emotion_sec_task = asyncio.create_task(
        query_emotion_model(
            emotion_wav_b64,
            mode="sec",
            base_url=cfg.emotion_vllm_base_url,
            model_name=cfg.emotion_vllm_model_name,
            timeout=cfg.emotion_request_timeout,
            max_tokens=256,
        )
    )

    asr_out, emotion_ser_out, emotion_sec_out = await asyncio.gather(
        asr_task,
        emotion_ser_task,
        emotion_sec_task,
        return_exceptions=True,
    )
    if isinstance(asr_out, HTTPException):
        raise asr_out
    if isinstance(asr_out, Exception):
        logger.error("Audio analyze ASR failed: %s", asr_out)
        raise HTTPException(status_code=502, detail=str(asr_out)) from asr_out
    asr_result = asr_out

    if isinstance(emotion_ser_out, Exception):
        logger.error("Audio analyze SER inference failed: %s", emotion_ser_out)
    if isinstance(emotion_sec_out, Exception):
        logger.error("Audio analyze SEC inference failed: %s", emotion_sec_out)
    emotion_ser = _emotion_result_payload(
        emotion_ser_out,
        mode="ser",
        duration_sec=emotion_duration_sec,
        language=language,
    )
    emotion_sec = _emotion_result_payload(
        emotion_sec_out,
        mode="sec",
        duration_sec=emotion_duration_sec,
        language=language,
    )
    emotion_payload = {
        "type": "final_emotion_pair",
        "mode": "both",
        "ser": emotion_ser,
        "sec": emotion_sec,
    }

    try:
        cleaned = await clean_asr_text(
            str(asr_result.get("text") or ""),
            hotwords=[],
            language=str(asr_result.get("language") or language or ""),
            emotion=emotion_payload,
            cfg=cfg,
        )
    except TextCleanupConfigError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Audio analyze text cleanup failed")
        raise HTTPException(
            status_code=502,
            detail=f"text cleanup model failed: {exc}",
        ) from exc

    return {
        "type": "audio_analysis",
        "duration_sec": round(duration_sec, 3),
        "language": asr_result.get("language") or language or "",
        "hotwords": hw_list,
        "asr": _public_asr_payload(asr_result),
        "cleaned_asr": _public_cleanup_payload(cleaned),
        "emotion": emotion_payload,
    }


class _RevalidateStaticFiles(StaticFiles):
    """Static files with tiered caching.

    Browsers were aggressively caching ``app.js`` / ``style.css`` because
    starlette's default ``StaticFiles`` ships no explicit ``Cache-Control``
    header, leaving the heuristic up to the client. That made shipping
    frontend fixes during a session unreliable — users had to hard-reload
    to pick up changes. We inject ``no-cache`` so the browser still uses
    its disk copy, but always revalidates with the server's ``ETag`` (set
    by starlette from mtime+size); unchanged files come back as 304 so
    bandwidth stays cheap. Cache-Control is omitted on non-200 responses
    to avoid pinning errors.

    For the assets the demo loads on every page (CSS, JS) we additionally
    grant a short ``max-age`` so the browser can serve repeat requests
    from disk cache without a conditional round-trip. Ten seconds is long
    enough to cover all the navigations of a single user session yet
    short enough that any real frontend change still appears within a
    blink — and the ``ETag`` revalidation kicks back in once the window
    expires. HTML is intentionally left at plain ``no-cache`` so users
    always see the latest markup.
    """

    _CACHEABLE_EXTS = (".css", ".js")

    @staticmethod
    def _cache_header_for(path: str) -> bytes:
        if path.endswith(_RevalidateStaticFiles._CACHEABLE_EXTS):
            return b"no-cache, max-age=10"
        return b"no-cache"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return

        cache_header = self._cache_header_for((scope.get("path") or "").lower())

        async def send_with_cache(message: dict) -> None:
            if message["type"] == "http.response.start":
                status = message.get("status", 0)
                if 200 <= status < 300:
                    headers = list(message.get("headers", []))
                    headers = [
                        (k, v) for (k, v) in headers if k.lower() != b"cache-control"
                    ]
                    headers.append((b"cache-control", cache_header))
                    message["headers"] = headers
            await send(message)

        await super().__call__(scope, receive, send_with_cache)


# Static mount comes last so it doesn't shadow the /api routes above.
app.mount(
    "/", _RevalidateStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"
)
