import asyncio
import json
import logging
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .asr.client import query_audio_model, query_audio_model_secondary
from .asr.fusion import choose_fused_result
from .asr.hotword import query_text_hotwords, sanitize_hotwords
from .audio.utils import Resampler48to16, pcm_to_wav_base64
from .audio.vad import VADProcessor
from .config import (
    DEBUG_SHOW_DUAL_ASR,
    EMOTION_MAX_AUDIO_SECONDS,
    EMOTION_REQUEST_TIMEOUT,
    EMOTION_VLLM_BASE_URL,
    EMOTION_VLLM_MODEL_NAME,
    ENABLE_PRIMARY_ASR,
    ENABLE_PSEUDO_STREAM,
    ENABLE_SECONDARY_ASR,
    MIN_SEGMENT_DURATION_MS,
    PRIMARY_ASR_TIMEOUT,
    PSEUDO_STREAM_INTERVAL_MS,
    SAMPLE_RATE,
)
from .emotion.client import query_emotion_model

logger = logging.getLogger(__name__)

MIN_SEGMENT_SAMPLES = int(SAMPLE_RATE * MIN_SEGMENT_DURATION_MS / 1000)

VALID_SRC_LANG = frozenset({"N/A", "Chinese", "English", "Indonesian", "Thai"})


def normalize_client_src_lang(value: object) -> str:
    s = str(value or "").strip()
    if not s:
        return "N/A"
    if s in VALID_SRC_LANG:
        return s
    logger.warning("Unknown src_lang %r, using N/A", s)
    return "N/A"


def _generate_segment_id() -> str:
    return f"seg-{int(time.time() * 1000)}"


class AudioSession:
    """Manages one WebSocket session: VAD ingestion + ASR pipeline."""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.segment_queue: asyncio.Queue[tuple | None] = asyncio.Queue(maxsize=20)
        self.vad = VADProcessor()
        self.resampler = Resampler48to16()
        self._pcm_carry = np.empty(0, dtype=np.float32)
        self.hotwords: list[str] = []
        self.src_lang: str = "N/A"
        self.enable_emotion: bool = False
        self.stop_event = asyncio.Event()
        self.extract_tasks: set[asyncio.Task] = set()
        self._ws_closed = False

        # Pseudo-streaming state
        self._utterance_id: str | None = None
        self._partial_seq: int = 0
        self._last_partial_time: float = 0.0
        self._partial_task: asyncio.Task | None = None
        self._pseudo_stream_interval: float = PSEUDO_STREAM_INTERVAL_MS / 1000.0

    async def _send_json(self, data: dict) -> bool:
        """Send JSON over WebSocket. Returns False if the connection is gone."""
        if self._ws_closed:
            return False
        try:
            await self.ws.send_json(data)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self._ws_closed = True
            self.stop_event.set()
            return False

    async def run(self) -> None:
        try:
            await asyncio.gather(self._vad_loop(), self._asr_loop())
        except Exception:
            logger.exception("Session error")

    async def cleanup(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        if self.extract_tasks:
            for task in self.extract_tasks:
                task.cancel()
            await asyncio.gather(*self.extract_tasks, return_exceptions=True)
        logger.info("Session ended")

    # ------------------------------------------------------------------
    # VAD loop: receive audio frames + control messages
    # ------------------------------------------------------------------

    async def _vad_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                msg = await self.ws.receive()

                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"]:
                    self._ingest_audio(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    self._handle_control_message(msg["text"])

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected (vad_loop)")
        finally:
            remaining = self.vad.flush()
            if remaining is not None and len(remaining) >= MIN_SEGMENT_SAMPLES:
                seg_id = self._utterance_id or _generate_segment_id()
                self._utterance_id = None
                await self.segment_queue.put(
                    (
                        seg_id,
                        remaining,
                        list(self.hotwords),
                        self.src_lang,
                    )
                )
            self.stop_event.set()
            await self.segment_queue.put(None)

    def _ingest_audio(self, raw_bytes: bytes) -> None:
        pcm_48k = (
            np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        pcm = self.resampler.process(pcm_48k)
        if pcm.size == 0:
            return
        if self._pcm_carry.size > 0:
            pcm = np.concatenate([self._pcm_carry, pcm])
        hop = self.vad.hop_size
        used = (len(pcm) // hop) * hop
        self._pcm_carry = pcm[used:].copy() if used < len(pcm) else np.empty(0, dtype=np.float32)
        for i in range(0, used, hop):
            segment = self.vad.process(pcm[i : i + hop])
            if segment is not None:
                self._enqueue_segment(segment)

        # --- Pseudo-streaming: manage utterance_id & throttled partial ---
        if not (ENABLE_PSEUDO_STREAM and ENABLE_SECONDARY_ASR):
            return

        if self.vad.is_speaking:
            if self._utterance_id is None:
                self._utterance_id = _generate_segment_id()
                self._partial_seq = 0
                self._last_partial_time = 0.0
                logger.debug("Utterance started: %s", self._utterance_id)

            now = time.monotonic()
            if now - self._last_partial_time >= self._pseudo_stream_interval:
                snapshot = self.vad.snapshot_incomplete_speech()
                if snapshot is not None and len(snapshot) >= MIN_SEGMENT_SAMPLES:
                    if self._partial_task is None or self._partial_task.done():
                        self._last_partial_time = now
                        self._partial_seq += 1
                        self._partial_task = asyncio.create_task(
                            self._emit_partial(
                                self._utterance_id,
                                snapshot,
                                self._partial_seq,
                            )
                        )
        else:
            if self._utterance_id is not None and not self.vad.is_speaking:
                self._utterance_id = None

    def _enqueue_segment(self, segment: np.ndarray) -> None:
        if len(segment) < MIN_SEGMENT_SAMPLES:
            logger.info(
                "Drop short segment (%.1fs < %.1fs)",
                len(segment) / SAMPLE_RATE,
                MIN_SEGMENT_DURATION_MS / 1000.0,
            )
            self._utterance_id = None
            return
        seg_id = self._utterance_id or _generate_segment_id()
        self._utterance_id = None
        try:
            self.segment_queue.put_nowait(
                (seg_id, segment, list(self.hotwords), self.src_lang)
            )
        except asyncio.QueueFull:
            logger.warning("Segment queue full, dropping %s", seg_id)

    def _handle_control_message(self, text: str) -> None:
        try:
            ctrl = json.loads(text)
        except json.JSONDecodeError:
            return

        if ctrl.get("type") == "update_hotwords":
            self.hotwords = sanitize_hotwords(ctrl.get("hotwords", []))
            if "src_lang" in ctrl:
                self.src_lang = normalize_client_src_lang(ctrl.get("src_lang"))
            if "enable_emotion" in ctrl:
                self.enable_emotion = bool(ctrl.get("enable_emotion"))
            logger.info(
                "Hotwords updated: %s (src_lang=%s, emotion=%s)",
                self.hotwords,
                self.src_lang,
                self.enable_emotion,
            )

        elif ctrl.get("type") == "update_emotion":
            self.enable_emotion = bool(ctrl.get("enabled"))
            logger.info("Emotion recognition toggled: %s", self.enable_emotion)

        elif ctrl.get("type") == "extract_hotwords":
            request_id = str(ctrl.get("request_id", "")).strip()
            source_text = str(ctrl.get("text", ""))
            task = asyncio.create_task(
                self._extract_hotwords(request_id, source_text)
            )
            self.extract_tasks.add(task)
            task.add_done_callback(self.extract_tasks.discard)

    async def _extract_hotwords(self, request_id: str, source_text: str) -> None:
        try:
            extracted = await query_text_hotwords(source_text)
            await self._send_json(
                {
                    "type": "extract_hotwords_result",
                    "request_id": request_id,
                    "hotwords": extracted,
                }
            )
        except WebSocketDisconnect:
            return
        except Exception as e:
            logger.exception(
                "extract_hotwords failed (request_id=%s)", request_id or "n/a"
            )
            await self._send_json(
                {
                    "type": "extract_hotwords_error",
                    "request_id": request_id,
                    "message": str(e),
                }
            )

    # ------------------------------------------------------------------
    # Pseudo-streaming: partial secondary ASR while user is speaking
    # ------------------------------------------------------------------

    async def _emit_partial(
        self, utterance_id: str, snapshot: np.ndarray, seq: int
    ) -> None:
        """Run both ASR models; use Secondary as silence gate, emit Primary text."""
        try:
            wav_b64 = pcm_to_wav_base64(snapshot)

            secondary_task = asyncio.create_task(
                query_audio_model_secondary(wav_b64)
            )
            primary_task = asyncio.create_task(
                query_audio_model(
                    wav_b64,
                    hotwords=list(self.hotwords),
                    src_lang=self.src_lang,
                )
            )

            secondary_res = await secondary_task
            secondary_text = str(
                (secondary_res or {}).get("transcription") or ""
            ).strip()
            if not secondary_text:
                primary_task.cancel()
                return

            primary_res = await primary_task
            text = str(
                (primary_res or {}).get("transcription") or ""
            ).strip()
            if not text:
                return

            await self._send_json(
                {
                    "type": "partial_transcript",
                    "utterance_id": utterance_id,
                    "text": text,
                    "seq": seq,
                }
            )
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("Partial ASR failed for %s seq=%s", utterance_id, seq)

    # ------------------------------------------------------------------
    # ASR loop: consume segments, query models, send results
    # ------------------------------------------------------------------

    async def _asr_loop(self) -> None:
        while True:
            item = await self.segment_queue.get()
            if item is None:
                break

            seg_id, segment, hw_snapshot, lang_snapshot = item
            logger.info(
                "Processing segment %s (%.1fs, hotwords=%s, src_lang=%s)",
                seg_id,
                len(segment) / SAMPLE_RATE,
                hw_snapshot,
                lang_snapshot,
            )

            try:
                await self._process_segment(
                    seg_id, segment, hw_snapshot, lang_snapshot
                )
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception as e:
                logger.exception("LLM query failed for %s", seg_id)
                await self._send_vad_event(seg_id, segment)
                if not await self._send_json(
                    {"type": "error", "id": seg_id, "message": str(e)}
                ):
                    break

    async def _process_segment(
        self,
        seg_id: str,
        segment: np.ndarray,
        hw_snapshot: list[str],
        lang_snapshot: str,
    ) -> None:
        wav_b64 = pcm_to_wav_base64(segment)
        primary_res: object = None
        secondary_res: object = None

        if ENABLE_SECONDARY_ASR:
            secondary_res, primary_res = await self._dual_asr_pipeline(
                seg_id, wav_b64, hw_snapshot, lang_snapshot
            )
            if secondary_res is None and primary_res is None:
                return
        else:
            if ENABLE_PRIMARY_ASR:
                primary_res = await asyncio.wait_for(
                    query_audio_model(
                        wav_b64,
                        hotwords=hw_snapshot,
                        src_lang=lang_snapshot,
                    ),
                    timeout=PRIMARY_ASR_TIMEOUT,
                )

        primary_result = None if isinstance(primary_res, Exception) else primary_res
        secondary_result = None if isinstance(secondary_res, Exception) else secondary_res

        if isinstance(primary_res, Exception):
            logger.warning("Primary ASR failed for %s: %s", seg_id, primary_res)
        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed for %s: %s", seg_id, secondary_res)
        if primary_result is None and secondary_result is None:
            raise RuntimeError("Both ASR models failed for this segment.")

        fused = choose_fused_result(primary_result, secondary_result, hotwords=hw_snapshot)

        if not str(fused.get("text") or "").strip():
            logger.info("Skip empty response for %s (silence)", seg_id)
            return

        await self._send_vad_event(seg_id, segment, wav_b64)

        payload: dict = {
            "type": "response",
            "id": seg_id,
            "text": fused["text"],
            "model_hotwords": fused["model_hotwords"],
        }
        if primary_result and primary_result.get("detected_language"):
            payload["src_lang_detected"] = primary_result["detected_language"]
        if DEBUG_SHOW_DUAL_ASR:
            payload.update(
                {
                    "text_primary": fused["primary_text"],
                    "text_secondary": fused["secondary_text"],
                    "fusion_meta": fused["fusion"],
                }
            )

        if self.enable_emotion:
            emotion_payload = await self._run_emotion(segment)
            if emotion_payload is not None:
                payload["emotion"] = emotion_payload

        await self._send_json(payload)

    async def _run_emotion(self, segment: np.ndarray) -> dict | None:
        """Run SER + SEC in parallel on the final segment. Best-effort."""
        audio_duration = len(segment) / SAMPLE_RATE
        clip = segment
        if (
            EMOTION_MAX_AUDIO_SECONDS > 0
            and audio_duration > EMOTION_MAX_AUDIO_SECONDS
        ):
            max_samples = int(SAMPLE_RATE * EMOTION_MAX_AUDIO_SECONDS)
            clip = segment[-max_samples:]

        try:
            wav_b64 = pcm_to_wav_base64(clip)
        except Exception:
            logger.exception("Emotion: failed to encode wav")
            return None

        async def _call(mode: str):
            return await query_emotion_model(
                wav_b64,
                mode=mode,
                base_url=EMOTION_VLLM_BASE_URL,
                model_name=EMOTION_VLLM_MODEL_NAME,
                timeout=EMOTION_REQUEST_TIMEOUT,
            )

        ser_res, sec_res = await asyncio.gather(
            _call("ser"), _call("sec"), return_exceptions=True
        )

        if isinstance(ser_res, Exception):
            logger.warning("SER inference failed: %s", ser_res)
            ser_res = None
        if isinstance(sec_res, Exception):
            logger.warning("SEC inference failed: %s", sec_res)
            sec_res = None

        ser_label = str((ser_res or {}).get("label", "") or "").strip()
        sec_text = str((sec_res or {}).get("text", "") or "").strip()
        sec_label = str((sec_res or {}).get("label", "") or "").strip()

        if not ser_label and not sec_text and not sec_label:
            return None

        return {
            "ser_label": ser_label,
            "sec_text": sec_text,
            "sec_label": sec_label,
        }

    async def _send_vad_event(
        self,
        seg_id: str,
        segment: np.ndarray,
        wav_b64: str | None = None,
    ) -> None:
        if wav_b64 is None:
            wav_b64 = pcm_to_wav_base64(segment)
        await self._send_json(
            {
                "type": "vad_event",
                "event": "segment_detected",
                "id": seg_id,
                "duration": f"{len(segment) / SAMPLE_RATE:.1f}s",
                "audio_b64": wav_b64,
            }
        )

    async def _dual_asr_pipeline(
        self,
        seg_id: str,
        wav_b64: str,
        hw_snapshot: list[str],
        lang_snapshot: str,
    ) -> tuple:
        """Run both ASR models in parallel, wait for both, return results.

        Returns (secondary_res, primary_res).  Returns (None, None) when the
        segment is silence.
        """
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(wav_b64, hotwords=hw_snapshot)
        )
        primary_task = None
        if ENABLE_PRIMARY_ASR:
            primary_task = asyncio.create_task(
                asyncio.wait_for(
                    query_audio_model(
                        wav_b64,
                        hotwords=hw_snapshot,
                        src_lang=lang_snapshot,
                    ),
                    timeout=PRIMARY_ASR_TIMEOUT,
                )
            )

        secondary_res = await secondary_task
        primary_res: object = None

        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed for %s: %s", seg_id, secondary_res)
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
            logger.info("Skip empty response for %s (secondary silence)", seg_id)
            if primary_task is not None:
                primary_task.cancel()
            return None, None

        if primary_task is not None:
            try:
                primary_res = await primary_task
            except Exception as err:
                primary_res = err

        return secondary_res, primary_res
