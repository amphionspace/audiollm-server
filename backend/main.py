import asyncio
import difflib
import json
import logging
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .audio_utils import pcm_to_wav_base64
from .config import (
    DEBUG_SHOW_DUAL_ASR,
    ENABLE_PRIMARY_ASR,
    ENABLE_SECONDARY_ASR,
    FUSION_DISAGREEMENT_THRESHOLD,
    FUSION_HOTWORD_BOOST,
    FUSION_MAX_REPETITION_RATIO,
    FUSION_MIN_PRIMARY_SCORE,
    FUSION_PRIMARY_SCORE_MARGIN,
    FUSION_SIMILARITY_THRESHOLD,
    MIN_SEGMENT_DURATION_MS,
    PRIMARY_ASR_TIMEOUT,
    SAMPLE_RATE,
)
from .llm_client import (
    close_client,
    query_audio_model,
    query_audio_model_secondary,
    query_text_hotwords,
)
from .vad_processor import VADProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio LLM Demo")
HOTWORD_LIMIT = 30
MIN_SEGMENT_SAMPLES = int(SAMPLE_RATE * MIN_SEGMENT_DURATION_MS / 1000)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.on_event("shutdown")
async def shutdown_event():
    await close_client()


def _generate_segment_id() -> str:
    return f"seg-{int(time.time() * 1000)}"


def _sanitize_hotwords(words) -> list[str]:
    if not isinstance(words, list):
        return []
    cleaned: list[str] = []
    for item in words:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in cleaned:
            continue
        cleaned.append(value)
        if len(cleaned) >= HOTWORD_LIMIT:
            break
    return cleaned


def _normalize_text_for_similarity(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFKC", raw).strip().lower()
    normalized = normalized.replace("，", ",").replace("。", ".").replace("；", ";")
    normalized = normalized.replace("：", ":").replace("？", "?").replace("！", "!")
    normalized = re.sub(r"[`~^*_=+|\\]+", " ", normalized)
    normalized = re.sub(r"\b(um+|uh+|emm+)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize_for_metrics(text: str) -> list[str]:
    normalized = _normalize_text_for_similarity(text)
    return [token for token in re.split(r"[\s,.;:!?]+", normalized) if token]


def _repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    unique_tokens = len(set(tokens))
    repeated = max(0, len(tokens) - unique_tokens)
    return repeated / len(tokens)


def _longest_run_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    max_run = 1
    current_run = 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run / len(tokens)


def _abnormal_char_ratio(text: str) -> float:
    normalized = _normalize_text_for_similarity(text)
    if not normalized:
        return 1.0
    allowed = 0
    for ch in normalized:
        if ch.isalnum() or ch in {" ", ",", ".", ";", ":", "!", "?", "'", '"', "-", "/"}:
            allowed += 1
    return 1.0 - (allowed / len(normalized))


def _hotword_hit_count(text: str, hotwords: list[str]) -> int:
    normalized_text = _normalize_text_for_similarity(text)
    hits = 0
    for hotword in hotwords:
        hw_norm = _normalize_text_for_similarity(hotword)
        if hw_norm and hw_norm in normalized_text:
            hits += 1
    return hits


def _quality_score(text: str, hotwords: list[str], hotword_boost: float) -> dict[str, float]:
    normalized = _normalize_text_for_similarity(text)
    if not normalized:
        return {
            "score": 0.0,
            "repetition_ratio": 1.0,
            "longest_run_ratio": 1.0,
            "abnormal_char_ratio": 1.0,
            "hotword_hits": 0.0,
        }

    tokens = _tokenize_for_metrics(normalized)
    repetition = _repetition_ratio(tokens)
    longest_run = _longest_run_ratio(tokens)
    abnormal_char = _abnormal_char_ratio(normalized)
    hotword_hits = _hotword_hit_count(normalized, hotwords)

    score = 1.0
    score -= min(0.45, repetition * 0.7)
    score -= min(0.35, longest_run * 0.6)
    score -= min(0.30, abnormal_char * 0.5)
    score += min(0.35, hotword_hits * hotword_boost)
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "repetition_ratio": round(repetition, 4),
        "longest_run_ratio": round(longest_run, 4),
        "abnormal_char_ratio": round(abnormal_char, 4),
        "hotword_hits": float(hotword_hits),
    }


def _text_similarity(a: str, b: str) -> float:
    left = _normalize_text_for_similarity(a)
    right = _normalize_text_for_similarity(b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _choose_fused_result(
    primary_result: dict[str, str | list[str]] | None,
    secondary_result: dict[str, str | list[str]] | None,
    hotwords: list[str],
) -> dict[str, object]:
    primary_text = str((primary_result or {}).get("transcription") or "")
    secondary_text = str((secondary_result or {}).get("transcription") or "")
    primary_hotwords = list((primary_result or {}).get("reported_hotwords") or [])
    secondary_hotwords = list((secondary_result or {}).get("reported_hotwords") or [])
    normalized_primary = _normalize_text_for_similarity(primary_text)
    normalized_secondary = _normalize_text_for_similarity(secondary_text)

    # If secondary ASR returns empty, always treat as silence.
    # This is an explicit product rule to avoid false positives from primary-only output.
    if not secondary_text:
        return {
            "text": "",
            "model_hotwords": [],
            "primary_text": primary_text,
            "secondary_text": secondary_text,
            "fusion": {
                "selected": "silence",
                "reason": "secondary_empty_force_silence",
                "similarity": None,
                "scores": {"primary": 0.0, "secondary": 0.0},
                "normalized_preview": {
                    "primary": normalized_primary,
                    "secondary": normalized_secondary,
                },
            },
        }

    if secondary_text and not primary_text:
        return {
            "text": secondary_text,
            "model_hotwords": secondary_hotwords,
            "primary_text": primary_text,
            "secondary_text": secondary_text,
            "fusion": {
                "selected": "secondary_only",
                "reason": "primary_empty",
                "similarity": None,
                "scores": {"primary": 0.0, "secondary": 1.0},
                "normalized_preview": {
                    "primary": normalized_primary,
                    "secondary": normalized_secondary,
                },
            },
        }

    if not primary_text and not secondary_text:
        return {
            "text": "",
            "model_hotwords": [],
            "primary_text": "",
            "secondary_text": "",
            "fusion": {
                "selected": "empty",
                "reason": "both_empty",
                "similarity": None,
                "scores": {"primary": 0.0, "secondary": 0.0},
                "normalized_preview": {"primary": "", "secondary": ""},
            },
        }

    similarity = _text_similarity(primary_text, secondary_text)
    primary_metrics = _quality_score(primary_text, hotwords, FUSION_HOTWORD_BOOST)
    secondary_metrics = _quality_score(secondary_text, hotwords, 0.0)
    primary_score = float(primary_metrics["score"])
    secondary_score = float(secondary_metrics["score"])
    primary_repetition = float(primary_metrics["repetition_ratio"])
    primary_hotword_hits = int(primary_metrics["hotword_hits"])
    disagreement = 1.0 - similarity

    primary_is_hallucination_risk = (
        primary_repetition > FUSION_MAX_REPETITION_RATIO
        or disagreement > FUSION_DISAGREEMENT_THRESHOLD
        and primary_metrics["hotword_hits"] <= secondary_metrics["hotword_hits"]
    )
    primary_meets_bar = primary_score >= FUSION_MIN_PRIMARY_SCORE
    primary_better = primary_score >= (secondary_score + FUSION_PRIMARY_SCORE_MARGIN)

    # Product rule: if primary explicitly hits configured hotwords, prefer it.
    if primary_hotword_hits > 0:
        selected = "primary_hotword_hit"
        reason = "primary_hits_hotword"
        selected_text = primary_text
        selected_hotwords = primary_hotwords
    elif primary_is_hallucination_risk and secondary_text:
        selected = "secondary_qwen_fallback"
        reason = "primary_hallucination_risk"
        selected_text = secondary_text
        selected_hotwords = secondary_hotwords
    elif similarity >= FUSION_SIMILARITY_THRESHOLD and primary_meets_bar:
        selected = "primary_agreement"
        reason = "high_similarity_and_primary_valid"
        selected_text = primary_text
        selected_hotwords = primary_hotwords
    elif primary_better and primary_meets_bar:
        selected = "primary_hotword_advantage"
        reason = "primary_score_margin"
        selected_text = primary_text
        selected_hotwords = primary_hotwords
    else:
        selected = "secondary_qwen_fallback"
        reason = "primary_not_confident"
        selected_text = secondary_text
        selected_hotwords = secondary_hotwords

    return {
        "text": selected_text,
        "model_hotwords": selected_hotwords,
        "primary_text": primary_text,
        "secondary_text": secondary_text,
        "fusion": {
            "selected": selected,
            "reason": reason,
            "similarity": round(similarity, 4),
            "threshold": FUSION_SIMILARITY_THRESHOLD,
            "disagreement": round(disagreement, 4),
            "scores": {
                "primary": round(primary_score, 4),
                "secondary": round(secondary_score, 4),
            },
            "metrics": {
                "primary": primary_metrics,
                "secondary": secondary_metrics,
            },
            "normalized_preview": {
                "primary": normalized_primary,
                "secondary": normalized_secondary,
            },
        },
    }


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    segment_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
    vad = VADProcessor()
    hotwords: list[str] = []
    stop_event = asyncio.Event()
    extract_tasks: set[asyncio.Task] = set()

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
                    hop = vad.hop_size
                    for i in range(0, len(pcm), hop):
                        frame = pcm[i : i + hop]
                        if len(frame) < hop:
                            break
                        segment = vad.process(frame)
                        if segment is not None:
                            if len(segment) < MIN_SEGMENT_SAMPLES:
                                logger.info(
                                    "Drop short segment (%.1fs < %.1fs)",
                                    len(segment) / SAMPLE_RATE,
                                    MIN_SEGMENT_DURATION_MS / 1000.0,
                                )
                                continue
                            seg_id = _generate_segment_id()
                            try:
                                await websocket.send_json(
                                    {
                                        "type": "vad_event",
                                        "event": "segment_detected",
                                        "id": seg_id,
                                        "duration": f"{len(segment) / SAMPLE_RATE:.1f}s",
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
                            hotwords = _sanitize_hotwords(ctrl.get("hotwords", []))
                            logger.info("Hotwords updated: %s", hotwords)
                        elif ctrl.get("type") == "extract_hotwords":
                            request_id = str(ctrl.get("request_id", "")).strip()
                            text = str(ctrl.get("text", ""))

                            async def extract_hotwords_task(
                                req_id: str = request_id,
                                source_text: str = text,
                            ):
                                try:
                                    extracted = await query_text_hotwords(source_text)
                                    await websocket.send_json(
                                        {
                                            "type": "extract_hotwords_result",
                                            "request_id": req_id,
                                            "hotwords": extracted,
                                        }
                                    )
                                except WebSocketDisconnect:
                                    return
                                except Exception as e:
                                    logger.exception(
                                        "extract_hotwords failed (request_id=%s)",
                                        req_id or "n/a",
                                    )
                                    try:
                                        await websocket.send_json(
                                            {
                                                "type": "extract_hotwords_error",
                                                "request_id": req_id,
                                                "message": str(e),
                                            }
                                        )
                                    except Exception:
                                        return

                            task = asyncio.create_task(extract_hotwords_task())
                            extract_tasks.add(task)
                            task.add_done_callback(extract_tasks.discard)
                    except json.JSONDecodeError:
                        pass

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected (vad_task)")
        finally:
            remaining = vad.flush()
            if remaining is not None and len(remaining) >= MIN_SEGMENT_SAMPLES:
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
                len(segment) / SAMPLE_RATE,
                hw_snapshot,
            )

            try:
                await websocket.send_json(
                    {"type": "status", "id": seg_id, "status": "processing"}
                )
            except Exception:
                break

            try:
                wav_b64 = pcm_to_wav_base64(segment)
                primary_res = None
                secondary_res = None

                if ENABLE_SECONDARY_ASR:
                    secondary_task = asyncio.create_task(
                        query_audio_model_secondary(wav_b64, hotwords=hw_snapshot)
                    )
                    primary_task = None
                    if ENABLE_PRIMARY_ASR:
                        # Launch primary in parallel but don't block first response on it.
                        primary_task = asyncio.create_task(
                            asyncio.wait_for(
                                query_audio_model(wav_b64, hotwords=hw_snapshot),
                                timeout=PRIMARY_ASR_TIMEOUT,
                            )
                        )

                    # Stage 1: fast path, respond with secondary first.
                    secondary_res = await secondary_task
                    secondary_result = (
                        None if isinstance(secondary_res, Exception) else secondary_res
                    )
                    if isinstance(secondary_res, Exception):
                        logger.warning(
                            "Secondary ASR failed for %s: %s", seg_id, secondary_res
                        )
                        # Secondary endpoint may be temporarily unavailable.
                        # Fallback to primary instead of failing the whole segment.
                        secondary_result = None
                        secondary_res = None
                        if primary_task is not None:
                            try:
                                primary_res = await primary_task
                            except Exception as primary_err:
                                primary_res = primary_err
                        if primary_res is None or isinstance(primary_res, Exception):
                            raise RuntimeError(
                                "Both ASR models failed for this segment."
                            )
                        # No low-latency secondary response available; continue to fusion
                        # stage so primary can be returned.
                        secondary_text = ""
                    else:
                        secondary_text = str(
                            (secondary_result or {}).get("transcription") or ""
                        ).strip()
                    if not secondary_text:
                        logger.info("Skip empty response for %s (secondary silence)", seg_id)
                        if primary_task is not None and secondary_res is not None:
                            primary_task.cancel()
                        # If secondary returned empty but primary is available, keep the
                        # explicit product behavior (silence). If secondary is down, do
                        # not discard here; primary fallback should proceed.
                        if secondary_res is not None:
                            await websocket.send_json(
                                {"type": "discard", "id": seg_id, "reason": "silence"}
                            )
                            continue

                    early_payload = {
                        "type": "response",
                        "id": seg_id,
                        "text": secondary_text,
                        "model_hotwords": list(
                            (secondary_result or {}).get("reported_hotwords") or []
                        ),
                    }
                    if DEBUG_SHOW_DUAL_ASR:
                        early_payload.update(
                            {
                                "text_primary": "",
                                "text_secondary": secondary_text,
                                "fusion_meta": {
                                    "selected": "secondary_early",
                                    "reason": "low_latency_first_response",
                                },
                            }
                        )
                    await websocket.send_json(early_payload)

                    # Stage 2: optional refinement with primary.
                    if primary_task is not None:
                        try:
                            primary_res = await primary_task
                        except Exception as primary_err:
                            primary_res = primary_err
                else:
                    if ENABLE_PRIMARY_ASR:
                        primary_res = await asyncio.wait_for(
                            query_audio_model(wav_b64, hotwords=hw_snapshot),
                            timeout=PRIMARY_ASR_TIMEOUT,
                        )

                primary_result = (
                    None if isinstance(primary_res, Exception) else primary_res
                )
                secondary_result = (
                    None if isinstance(secondary_res, Exception) else secondary_res
                )

                if isinstance(primary_res, Exception):
                    logger.warning("Primary ASR failed for %s: %s", seg_id, primary_res)
                if isinstance(secondary_res, Exception):
                    logger.warning("Secondary ASR failed for %s: %s", seg_id, secondary_res)
                if primary_result is None and secondary_result is None:
                    raise RuntimeError("Both ASR models failed for this segment.")

                fused = _choose_fused_result(
                    primary_result, secondary_result, hotwords=hw_snapshot
                )
                # Silence: notify frontend to remove pending bubbles.
                if not str(fused.get("text") or "").strip():
                    logger.info("Skip empty response for %s (silence)", seg_id)
                    await websocket.send_json(
                        {"type": "discard", "id": seg_id, "reason": "silence"}
                    )
                    continue

                # If we already sent secondary fast-path and fusion keeps same text,
                # avoid redundant websocket updates in non-debug mode.
                if (
                    ENABLE_SECONDARY_ASR
                    and not DEBUG_SHOW_DUAL_ASR
                    and str(fused["text"]).strip()
                    == str((secondary_result or {}).get("transcription") or "").strip()
                ):
                    continue
                payload = {
                    "type": "response",
                    "id": seg_id,
                    "text": fused["text"],
                    "model_hotwords": fused["model_hotwords"],
                }
                if DEBUG_SHOW_DUAL_ASR:
                    payload.update(
                        {
                            "text_primary": fused["primary_text"],
                            "text_secondary": fused["secondary_text"],
                            "fusion_meta": fused["fusion"],
                        }
                    )
                await websocket.send_json(payload)
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
        if extract_tasks:
            for task in extract_tasks:
                task.cancel()
            await asyncio.gather(*extract_tasks, return_exceptions=True)
        logger.info("Session ended")


# Serve frontend static files (must be mounted last)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
