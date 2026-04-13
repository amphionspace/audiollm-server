import asyncio
import json
import logging
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .asr_client import query_audio_model, query_audio_model_secondary
from .audio_utils import pcm_to_wav_base64
from .config import (
    ENABLE_PRIMARY_ASR,
    ENABLE_PSEUDO_STREAM,
    ENABLE_SECONDARY_ASR,
    MIN_SEGMENT_DURATION_MS,
    PRIMARY_ASR_TIMEOUT,
    PSEUDO_STREAM_INTERVAL_MS,
    SAMPLE_RATE,
)
from .fusion import choose_fused_result
from .hotword_service import sanitize_hotwords
from .vad_processor import VADProcessor

logger = logging.getLogger(__name__)

MIN_SEGMENT_SAMPLES = int(SAMPLE_RATE * MIN_SEGMENT_DURATION_MS / 1000)

LANG_CODE_MAP: dict[str, str] = {
    "zh": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "id": "Indonesian",
    "th": "Thai",
}


def _map_language(lang_query: str) -> str:
    code = lang_query.strip().lower()
    if not code:
        return "N/A"
    if code in LANG_CODE_MAP:
        return LANG_CODE_MAP[code]
    for full_name in ("Chinese", "English", "Indonesian", "Thai"):
        if code == full_name.lower():
            return full_name
    return "N/A"


_SENTINEL = object()


class ASRStreamingSession:
    """tiro_api-compatible ASR WebSocket session.

    Protocol:
      1. Server sends ``ready``
      2. Client sends ``start`` (mode, format, sample_rate_hz, channels)
      3. Client sends binary PCM chunks (16 kHz s16le mono)
         - Server emits ``partial_asr`` while VAD detects speech
      4. Client sends ``stop``
         - Server flushes VAD, emits final ``final_asr``, closes
    """

    def __init__(self, websocket: WebSocket, language: str) -> None:
        self.ws = websocket
        self.language = language
        self.src_lang = _map_language(language)
        self.hotwords: list[str] = []

        self.vad = VADProcessor()
        self._pcm_carry = np.empty(0, dtype=np.float32)

        self._work_queue: asyncio.Queue = asyncio.Queue(maxsize=40)

        self._partial_interval = PSEUDO_STREAM_INTERVAL_MS / 1000.0
        self._last_partial_time: float = 0.0
        self._partial_task: asyncio.Task | None = None

        self._started = False
        self._stopped = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self.ws.send_json({"type": "ready"})
        logger.info("Sent ready (language=%s, src_lang=%s)", self.language, self.src_lang)
        try:
            await asyncio.gather(self._receive_loop(), self._asr_loop())
        except Exception:
            logger.exception("ASRStreamingSession error")

    async def cleanup(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        logger.info("ASRStreamingSession ended")

    # ------------------------------------------------------------------
    # Receive loop: start / PCM / stop
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        try:
            while True:
                msg = await self.ws.receive()

                if msg.get("type") == "websocket.disconnect":
                    break

                if "text" in msg and msg["text"]:
                    ctrl = self._parse_json(msg["text"])
                    if ctrl is None:
                        continue
                    msg_type = ctrl.get("type", "")

                    if msg_type == "start":
                        self._handle_start(ctrl)
                    elif msg_type == "stop":
                        self._handle_stop()
                        break
                    elif msg_type == "update_hotwords":
                        self._handle_update_hotwords(ctrl)
                    else:
                        logger.debug("Ignoring unknown control message: %s", msg_type)

                elif "bytes" in msg and msg["bytes"]:
                    if not self._started or self._stopped:
                        continue
                    self._ingest_pcm(msg["bytes"])

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected (receive_loop)")
        finally:
            if not self._stopped:
                self._flush_vad()
            await self._work_queue.put(_SENTINEL)

    def _parse_json(self, text: str) -> dict | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from client: %s", text[:200])
            return None

    def _handle_start(self, ctrl: dict) -> None:
        if self._started:
            logger.warning("Duplicate start message, ignoring")
            return
        self._started = True
        fmt = ctrl.get("format", "pcm_s16le")
        sr = ctrl.get("sample_rate_hz", 16000)
        ch = ctrl.get("channels", 1)
        logger.info("Start received: mode=%s format=%s sr=%s ch=%s",
                     ctrl.get("mode"), fmt, sr, ch)

    def _handle_update_hotwords(self, ctrl: dict) -> None:
        self.hotwords = sanitize_hotwords(ctrl.get("hotwords", []))
        if "src_lang" in ctrl:
            lang_val = str(ctrl.get("src_lang", "")).strip()
            if lang_val:
                self.src_lang = _map_language(lang_val)
        logger.info("Hotwords updated: %s (src_lang=%s)", self.hotwords, self.src_lang)

    def _handle_stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        logger.info("Stop received, flushing VAD")
        self._flush_vad()

    # ------------------------------------------------------------------
    # PCM ingestion + VAD
    # ------------------------------------------------------------------

    def _ingest_pcm(self, raw_bytes: bytes) -> None:
        pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if self._pcm_carry.size > 0:
            pcm = np.concatenate([self._pcm_carry, pcm])

        hop = self.vad.hop_size
        used = (len(pcm) // hop) * hop
        self._pcm_carry = pcm[used:].copy() if used < len(pcm) else np.empty(0, dtype=np.float32)

        for i in range(0, used, hop):
            segment = self.vad.process(pcm[i : i + hop])
            if segment is not None:
                self._enqueue_final(segment)

        if ENABLE_PSEUDO_STREAM and (ENABLE_PRIMARY_ASR or ENABLE_SECONDARY_ASR) and self.vad.is_speaking:
            now = time.monotonic()
            if now - self._last_partial_time >= self._partial_interval:
                snapshot = self.vad.snapshot_incomplete_speech()
                if snapshot is not None and len(snapshot) >= MIN_SEGMENT_SAMPLES:
                    if self._partial_task is None or self._partial_task.done():
                        self._last_partial_time = now
                        self._partial_task = asyncio.create_task(
                            self._emit_partial(snapshot)
                        )

    def _enqueue_final(self, segment: np.ndarray) -> None:
        if len(segment) < MIN_SEGMENT_SAMPLES:
            logger.info("Drop short segment (%.1fs)", len(segment) / SAMPLE_RATE)
            return
        try:
            self._work_queue.put_nowait(("final", segment, list(self.hotwords)))
        except asyncio.QueueFull:
            logger.warning("Work queue full, dropping final segment")

    def _flush_vad(self) -> None:
        remaining = self.vad.flush()
        if remaining is not None and len(remaining) >= MIN_SEGMENT_SAMPLES:
            try:
                self._work_queue.put_nowait(("final", remaining, list(self.hotwords)))
            except asyncio.QueueFull:
                logger.warning("Work queue full, dropping flushed segment")

    # ------------------------------------------------------------------
    # ASR loop: process segments from queue
    # ------------------------------------------------------------------

    async def _asr_loop(self) -> None:
        while True:
            item = await self._work_queue.get()
            if item is _SENTINEL:
                break
            kind, segment, hw_snapshot = item
            try:
                if kind == "final":
                    await self._process_final(segment, hw_snapshot)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.exception("ASR failed")
                try:
                    await self.ws.send_json({"type": "error", "message": str(e)})
                except Exception:
                    break

    async def _process_final(self, segment: np.ndarray, hw_snapshot: list[str]) -> None:
        audio_duration = len(segment) / SAMPLE_RATE
        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(segment)
        primary_res: object = None
        secondary_res: object = None

        if ENABLE_SECONDARY_ASR:
            secondary_res, primary_res = await self._dual_asr(wav_b64, hw_snapshot)
            if secondary_res is None and primary_res is None:
                return
        elif ENABLE_PRIMARY_ASR:
            primary_res = await asyncio.wait_for(
                query_audio_model(
                    wav_b64, hotwords=hw_snapshot, src_lang=self.src_lang,
                ),
                timeout=PRIMARY_ASR_TIMEOUT,
            )

        primary_result = None if isinstance(primary_res, Exception) else primary_res
        secondary_result = None if isinstance(secondary_res, Exception) else secondary_res

        if isinstance(primary_res, Exception):
            logger.warning("Primary ASR failed: %s", primary_res)
        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed: %s", secondary_res)
        if primary_result is None and secondary_result is None:
            raise RuntimeError("Both ASR models failed for this segment.")

        fused = choose_fused_result(primary_result, secondary_result, hotwords=hw_snapshot)
        text = str(fused.get("text") or "").strip()

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Final ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration, elapsed, rtf, text[:80],
        )

        if not text:
            return

        detected_lang = self.language
        if primary_result and primary_result.get("detected_language"):
            detected_lang = primary_result["detected_language"]

        await self.ws.send_json({
            "type": "final_asr",
            "text": text,
            "language": detected_lang,
        })

    async def _dual_asr(self, wav_b64: str, hw_snapshot: list[str]) -> tuple:
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(wav_b64, hotwords=hw_snapshot)
        )
        primary_task = None
        if ENABLE_PRIMARY_ASR:
            primary_task = asyncio.create_task(
                asyncio.wait_for(
                    query_audio_model(
                        wav_b64, hotwords=hw_snapshot, src_lang=self.src_lang,
                    ),
                    timeout=PRIMARY_ASR_TIMEOUT,
                )
            )

        secondary_res = await secondary_task
        primary_res: object = None

        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed: %s", secondary_res)
            secondary_res = None
            if primary_task is not None:
                try:
                    primary_res = await primary_task
                except Exception as err:
                    primary_res = err
            if primary_res is None or isinstance(primary_res, Exception):
                raise RuntimeError("Both ASR models failed for this segment.")
            return secondary_res, primary_res

        secondary_text = str(
            (secondary_res or {}).get("transcription") or ""
        ).strip()

        if not secondary_text:
            if primary_task is not None:
                primary_task.cancel()
            return None, None

        if primary_task is not None:
            try:
                primary_res = await primary_task
            except Exception as err:
                primary_res = err

        return secondary_res, primary_res

    # ------------------------------------------------------------------
    # Partial ASR (pseudo-streaming via VAD snapshot, dual-model fused)
    # ------------------------------------------------------------------

    async def _emit_partial(self, snapshot: np.ndarray) -> None:
        try:
            audio_duration = len(snapshot) / SAMPLE_RATE
            t0 = time.monotonic()
            wav_b64 = pcm_to_wav_base64(snapshot)
            hw_snapshot = list(self.hotwords)

            primary_res: object = None
            secondary_res: object = None

            if ENABLE_SECONDARY_ASR and ENABLE_PRIMARY_ASR:
                secondary_res, primary_res = await self._dual_asr(wav_b64, hw_snapshot)
                if secondary_res is None and primary_res is None:
                    return
            elif ENABLE_PRIMARY_ASR:
                primary_res = await asyncio.wait_for(
                    query_audio_model(
                        wav_b64, hotwords=hw_snapshot, src_lang=self.src_lang,
                    ),
                    timeout=PRIMARY_ASR_TIMEOUT,
                )
            elif ENABLE_SECONDARY_ASR:
                secondary_res = await query_audio_model_secondary(wav_b64, hotwords=hw_snapshot)

            primary_result = None if isinstance(primary_res, Exception) else primary_res
            secondary_result = None if isinstance(secondary_res, Exception) else secondary_res

            if primary_result is None and secondary_result is None:
                return

            if ENABLE_SECONDARY_ASR:
                sec_text = str((secondary_result or {}).get("transcription") or "").strip()
                if not sec_text:
                    logger.debug("Partial suppressed: secondary output empty (noise gate)")
                    return

            fused = choose_fused_result(primary_result, secondary_result, hotwords=hw_snapshot)
            text = str(fused.get("text") or "").strip()

            elapsed = time.monotonic() - t0
            rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
            logger.info(
                "Partial ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
                audio_duration, elapsed, rtf, text[:80],
            )

            if not text:
                return
            await self.ws.send_json({
                "type": "partial_asr",
                "text": text,
                "language": self.language,
            })
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("Partial ASR failed for snapshot", exc_info=True)
