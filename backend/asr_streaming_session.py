import asyncio
import json
import logging
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .asr.client import query_audio_model, query_audio_model_secondary
from .asr.fusion import choose_fused_result
from .asr.hotword import sanitize_hotwords
from .audio.utils import pcm_to_wav_base64
from .audio.vad import VADProcessor
from .config import SAMPLE_RATE, Config, load_config

logger = logging.getLogger(__name__)

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
    """ASR WebSocket session with per-session config override.

    Protocol:
      1. Server sends ``ready``
      2. Client sends ``start`` (format, sample_rate_hz, channels, language, hotwords, config)
      3. Client sends binary PCM chunks (16 kHz s16le mono)
         - Server emits ``partial`` while VAD detects speech
      4. Client sends ``stop``
         - Server flushes VAD, performs inference, emits ``final``
    """

    def __init__(self, websocket: WebSocket, language: str) -> None:
        self.ws = websocket
        self.language = language
        self.src_lang = _map_language(language)
        self.hotwords: list[str] = []

        self.cfg = load_config()
        self.vad = VADProcessor()
        self._pcm_carry = np.empty(0, dtype=np.float32)

        self._work_queue: asyncio.Queue = asyncio.Queue(maxsize=40)

        self._partial_interval = self.cfg.pseudo_stream_interval_ms / 1000.0
        self._last_partial_time: float = 0.0
        self._partial_task: asyncio.Task | None = None

        self._started = False
        self._stopped = False
        self._sent_final_after_stop = False

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
                self._flush_vad(force=True)
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

        # Client config override
        client_config = ctrl.get("config")
        if isinstance(client_config, dict) and client_config:
            self.cfg = self.cfg.override(**client_config)
            self._partial_interval = self.cfg.pseudo_stream_interval_ms / 1000.0
            logger.info("Config overridden by client: %s", list(client_config.keys()))

        # Language from start message
        lang_val = str(ctrl.get("language", "")).strip()
        if lang_val:
            self.language = lang_val
            self.src_lang = _map_language(lang_val)

        # Hotwords from start message
        hw_raw = ctrl.get("hotwords")
        if isinstance(hw_raw, list):
            self.hotwords = sanitize_hotwords(hw_raw)
            logger.info("Hotwords from start: %d items", len(self.hotwords))

        fmt = ctrl.get("format", "pcm_s16le")
        sr = ctrl.get("sample_rate_hz", 16000)
        ch = ctrl.get("channels", 1)
        logger.info("Start received: mode=%s format=%s sr=%s ch=%s language=%s",
                     ctrl.get("mode"), fmt, sr, ch, self.language)

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
        self._flush_vad(force=True)

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

        min_samples = int(SAMPLE_RATE * self.cfg.min_segment_duration_ms / 1000)

        for i in range(0, used, hop):
            segment = self.vad.process(pcm[i : i + hop])
            if segment is not None:
                self._enqueue_final(segment, min_samples)

        pseudo_ok = self.cfg.enable_pseudo_stream and (
            self.cfg.enable_primary_asr or self.cfg.enable_secondary_asr
        )
        if pseudo_ok and self.vad.is_speaking:
            now = time.monotonic()
            if now - self._last_partial_time >= self._partial_interval:
                snapshot = self.vad.snapshot_incomplete_speech()
                if snapshot is not None and len(snapshot) >= min_samples:
                    if self._partial_task is None or self._partial_task.done():
                        self._last_partial_time = now
                        self._partial_task = asyncio.create_task(
                            self._emit_partial(snapshot)
                        )

    def _enqueue_final(self, segment: np.ndarray, min_samples: int) -> None:
        if len(segment) < min_samples:
            logger.info("Drop short segment (%.1fs)", len(segment) / SAMPLE_RATE)
            return
        try:
            self._work_queue.put_nowait(("final", segment, list(self.hotwords)))
        except asyncio.QueueFull:
            logger.warning("Work queue full, dropping final segment")

    def _flush_vad(self, *, force: bool = False) -> None:
        """Flush remaining VAD audio into the work queue.

        When *force* is True (stop scenario), skip the minimum-duration check
        so that the remaining audio always gets processed.
        """
        remaining = self.vad.flush()
        if remaining is None or len(remaining) == 0:
            return
        min_samples = int(SAMPLE_RATE * self.cfg.min_segment_duration_ms / 1000)
        if not force and len(remaining) < min_samples:
            return
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
                    sent = await self._process_final(segment, hw_snapshot)
                    if sent and self._stopped:
                        self._sent_final_after_stop = True
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.exception("ASR failed")
                try:
                    await self.ws.send_json({"type": "error", "message": str(e)})
                except Exception:
                    break

        if self._stopped and not self._sent_final_after_stop:
            try:
                await self.ws.send_json({
                    "type": "final",
                    "text": "",
                    "language": self.language,
                })
            except Exception:
                pass

    async def _process_final(self, segment: np.ndarray, hw_snapshot: list[str]) -> bool:
        audio_duration = len(segment) / SAMPLE_RATE
        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(segment)
        primary_res: object = None
        secondary_res: object = None

        if self.cfg.enable_secondary_asr:
            secondary_res, primary_res = await self._dual_asr(wav_b64, hw_snapshot)
            if secondary_res is None and primary_res is None:
                return False
        elif self.cfg.enable_primary_asr:
            primary_res = await asyncio.wait_for(
                query_audio_model(
                    wav_b64, hotwords=hw_snapshot, src_lang=self.src_lang,
                    base_url=self.cfg.vllm_base_url,
                    model_name=self.cfg.vllm_model_name,
                    timeout=self.cfg.asr_request_timeout,
                ),
                timeout=self.cfg.primary_asr_timeout,
            )

        primary_result = None if isinstance(primary_res, Exception) else primary_res
        secondary_result = None if isinstance(secondary_res, Exception) else secondary_res

        if isinstance(primary_res, Exception):
            logger.warning("Primary ASR failed: %s", primary_res)
        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed: %s", secondary_res)
        if primary_result is None and secondary_result is None:
            raise RuntimeError("Both ASR models failed for this segment.")

        if primary_result and not secondary_result:
            text = str(primary_result.get("transcription") or "").strip()
            detected_lang = primary_result.get("detected_language") or self.language
        elif secondary_result and not primary_result:
            text = str(secondary_result.get("transcription") or "").strip()
            detected_lang = self.language
        else:
            fused = choose_fused_result(
                primary_result, secondary_result, hotwords=hw_snapshot,
                similarity_threshold=self.cfg.fusion_similarity_threshold,
                min_primary_score=self.cfg.fusion_min_primary_score,
                max_repetition_ratio=self.cfg.fusion_max_repetition_ratio,
                disagreement_threshold=self.cfg.fusion_disagreement_threshold,
                hotword_boost=self.cfg.fusion_hotword_boost,
                primary_score_margin=self.cfg.fusion_primary_score_margin,
            )
            text = str(fused.get("text") or "").strip()
            detected_lang = self.language
            if primary_result and primary_result.get("detected_language"):
                detected_lang = primary_result["detected_language"]

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Final ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration, elapsed, rtf, text[:80],
        )

        if not text:
            return False

        await self.ws.send_json({
            "type": "final",
            "text": text,
            "language": detected_lang,
        })
        return True

    async def _dual_asr(self, wav_b64: str, hw_snapshot: list[str]) -> tuple:
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(
                wav_b64, hotwords=hw_snapshot,
                base_url=self.cfg.secondary_vllm_base_url,
                model_name=self.cfg.secondary_vllm_model_name,
                timeout=self.cfg.asr_request_timeout,
            )
        )
        primary_task = None
        if self.cfg.enable_primary_asr:
            primary_task = asyncio.create_task(
                asyncio.wait_for(
                    query_audio_model(
                        wav_b64, hotwords=hw_snapshot, src_lang=self.src_lang,
                        base_url=self.cfg.vllm_base_url,
                        model_name=self.cfg.vllm_model_name,
                        timeout=self.cfg.asr_request_timeout,
                    ),
                    timeout=self.cfg.primary_asr_timeout,
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

            if self.cfg.enable_secondary_asr and self.cfg.enable_primary_asr:
                secondary_res, primary_res = await self._dual_asr(wav_b64, hw_snapshot)
                if secondary_res is None and primary_res is None:
                    return
            elif self.cfg.enable_primary_asr:
                primary_res = await asyncio.wait_for(
                    query_audio_model(
                        wav_b64, hotwords=hw_snapshot, src_lang=self.src_lang,
                        base_url=self.cfg.vllm_base_url,
                        model_name=self.cfg.vllm_model_name,
                        timeout=self.cfg.asr_request_timeout,
                    ),
                    timeout=self.cfg.primary_asr_timeout,
                )
            elif self.cfg.enable_secondary_asr:
                secondary_res = await query_audio_model_secondary(
                    wav_b64, hotwords=hw_snapshot,
                    base_url=self.cfg.secondary_vllm_base_url,
                    model_name=self.cfg.secondary_vllm_model_name,
                    timeout=self.cfg.asr_request_timeout,
                )

            primary_result = None if isinstance(primary_res, Exception) else primary_res
            secondary_result = None if isinstance(secondary_res, Exception) else secondary_res

            if primary_result is None and secondary_result is None:
                return

            if self.cfg.enable_secondary_asr:
                sec_text = str((secondary_result or {}).get("transcription") or "").strip()
                if not sec_text:
                    logger.debug("Partial suppressed: secondary output empty (noise gate)")
                    return

            if primary_result and not secondary_result:
                text = str(primary_result.get("transcription") or "").strip()
            elif secondary_result and not primary_result:
                text = str(secondary_result.get("transcription") or "").strip()
            else:
                fused = choose_fused_result(
                    primary_result, secondary_result, hotwords=hw_snapshot,
                    similarity_threshold=self.cfg.fusion_similarity_threshold,
                    min_primary_score=self.cfg.fusion_min_primary_score,
                    max_repetition_ratio=self.cfg.fusion_max_repetition_ratio,
                    disagreement_threshold=self.cfg.fusion_disagreement_threshold,
                    hotword_boost=self.cfg.fusion_hotword_boost,
                    primary_score_margin=self.cfg.fusion_primary_score_margin,
                )
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
                "type": "partial",
                "text": text,
                "language": self.language,
            })
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("Partial ASR failed for snapshot", exc_info=True)
