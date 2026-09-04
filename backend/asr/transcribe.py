"""Offline long-audio transcription pipeline.

Replays an entire decoded recording through the same
:class:`~backend.streaming.audio_stream.VadSegmentedStream` the live WS
endpoints use (so a given recording produces identical segment boundaries
whether it arrives as a stream or as a file), then runs the shared one-shot
dual-ASR inference per segment with bounded concurrency.

What the streaming stack does NOT provide and is added here:

- a max-segment force-cut: VAD only finalizes a segment on silence, so an
  uninterrupted monologue would otherwise grow without bound and exceed what
  a single model call can handle;
- parallel per-segment inference (a live session is inherently sequential,
  a file is not);
- single-retry + partial-failure semantics: one bad segment must not throw
  away an hour of meeting transcript.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..audio.utils import pcm_to_wav_base64
from ..clean_stream import (
    SessionOptions,
    evaluate_cleanup_result,
    infer_emotion,
    refine_text,
)
from ..config import SAMPLE_RATE, Config
from ..jobstore import JobExecutionError
from ..streaming.audio_stream import VadSegmentedStream
from ..streaming.events import SegmentReady
from .oneshot import run_oneshot_asr

logger = logging.getLogger(__name__)

# Replay granularity. Purely a CPU-loop batching knob (the VAD itself always
# steps in 10 ms hops); it also bounds how far past ``max_segment_sec`` a
# force-cut can overshoot (<= one chunk).
_FEED_CHUNK_SEC = 1.0

_BYTES_PER_SAMPLE = 2  # int16 mono


@dataclass
class OfflineSegment:
    """One VAD-cut speech segment positioned on the file's timeline."""

    index: int
    start_ms: float
    end_ms: float
    pcm: np.ndarray  # float32 16 kHz mono


def float_pcm_to_i16_bytes(pcm: np.ndarray) -> bytes:
    """float32 [-1, 1] -> int16 little-endian bytes (the stream feed format)."""
    return np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def segment_pcm_offline(
    pcm_i16: bytes,
    cfg: Config,
    *,
    max_segment_sec: float,
) -> list[OfflineSegment]:
    """Cut a whole recording into VAD speech segments with timeline stamps.

    ``pcm_i16`` is int16 LE mono 16 kHz bytes. Segments come back in file
    order; ``start_ms``/``end_ms`` are positions within the recording (same
    semantics as the streaming ``SegmentReady`` timing).

    ``max_segment_sec`` is the force-cut ceiling for uninterrupted speech:
    when the VAD's in-flight buffer exceeds it, the buffer is flushed as a
    finished segment and the state machine restarts. The cut lands on a feed
    chunk boundary, so segments may overshoot by up to ``_FEED_CHUNK_SEC``.
    The restart costs the next segment its pre-speech backfill and start-
    confirmation frames land in the new pre-speech buffer, so no audio is
    lost at the seam.

    ``transcribe_silence_duration_ms > 0`` swaps in an offline-only cut
    pause: live endpoints keep the latency-tuned global value while minutes
    transcripts get longer, more readable segments.
    """
    if cfg.transcribe_silence_duration_ms > 0:
        cfg = cfg.override(
            silence_duration_ms=cfg.transcribe_silence_duration_ms
        )
    stream = VadSegmentedStream(enable_partial=False)
    stream.configure(cfg)

    max_speech_samples = (
        int(max_segment_sec * SAMPLE_RATE) if max_segment_sec > 0 else 0
    )
    chunk_bytes = int(_FEED_CHUNK_SEC * SAMPLE_RATE) * _BYTES_PER_SAMPLE

    segments: list[OfflineSegment] = []

    def collect(events) -> None:
        for ev in events:
            if isinstance(ev, SegmentReady):
                segments.append(
                    OfflineSegment(
                        index=len(segments),
                        start_ms=float(ev.start_ms or 0.0),
                        end_ms=float(ev.end_ms or 0.0),
                        pcm=ev.pcm,
                    )
                )

    for off in range(0, len(pcm_i16), chunk_bytes):
        collect(stream.feed(pcm_i16[off : off + chunk_bytes]))
        if max_speech_samples and stream.vad.is_speaking:
            buffered = len(stream.vad.audio_buffer) * stream.vad.hop_size
            if buffered >= max_speech_samples:
                logger.info(
                    "Force-cutting continuous speech at %.1fs (cap %.0fs)",
                    buffered / SAMPLE_RATE,
                    max_segment_sec,
                )
                collect(stream.flush(force=True))

    collect(stream.flush(force=True))
    return segments


async def transcribe_pcm_i16(
    pcm_i16: bytes,
    *,
    cfg: Config,
    language: str = "",
    hotwords: list[str] | None = None,
    hotword_pool_id: str | None = None,
    enhancement_options: SessionOptions | None = None,
    on_segments_planned: Callable[[int], None] | None = None,
    on_segment_done: Callable[[int], None] | None = None,
    release_input: Callable[[], None] | None = None,
) -> dict:
    """Segment + transcribe a whole recording; returns the job result payload.

    Per-segment failures are retried once, then recorded on the segment entry
    (``error``) without failing the job; only "every segment failed" raises.
    Segments the model transcribes as empty (VAD-passed noise) are dropped
    from the output, mirroring the streaming engines' noise-skip behaviour.

    ``release_input`` is invoked right after segmentation so the caller can
    free the full-recording buffer while inference runs (segments hold their
    own copies).
    """
    hotwords = hotwords or []
    duration_sec = len(pcm_i16) / _BYTES_PER_SAMPLE / SAMPLE_RATE

    # CPU-bound segmentation takes ~13 s per half-hour of audio; run it in a
    # worker thread so it doesn't freeze the event loop (and with it every
    # live WS endpoint and concurrent job poll) for that long.
    segments = await asyncio.to_thread(
        segment_pcm_offline,
        pcm_i16,
        cfg,
        max_segment_sec=cfg.transcribe_max_segment_sec,
    )
    # Drop our local binding too, otherwise the caller's release_input()
    # can't actually free the full-recording buffer (segments own copies).
    del pcm_i16
    if release_input is not None:
        release_input()
    if on_segments_planned is not None:
        on_segments_planned(len(segments))
    logger.info(
        "Offline transcription: %.1fs audio -> %d segments", duration_sec, len(segments)
    )

    base_result: dict = {
        "type": "transcription",
        "language": language or "",
        "duration_sec": round(duration_sec, 3),
        "segments": [],
        "full_text": "",
        "failed_segments": 0,
    }
    if enhancement_options is not None:
        base_result.update(
            {
                "text": "",
                "builtin_hotword_lists": enhancement_options.builtin,
                "custom_hotword_count": len(enhancement_options.custom),
                "translate_mode": enhancement_options.translate_mode,
                "postprocess_mode": (
                    "translation" if enhancement_options.translate_mode else "cleanup"
                ),
            }
        )
        if enhancement_options.translate_mode:
            base_result["target_language"] = enhancement_options.target_language
        else:
            base_result["cleanup_level"] = enhancement_options.cleanup_level
    if not segments:
        if enhancement_options is not None and enhancement_options.text_emotion:
            base_result["emotion_bucket_count"] = 0
        if enhancement_options is not None and (
            enhancement_options.translate_mode
            or enhancement_options.cleanup_level != "off"
        ):
            status_key = (
                "translation_status"
                if enhancement_options.translate_mode
                else "cleanup_status"
            )
            text_key = (
                "translated_text"
                if enhancement_options.translate_mode
                else "cleaned_text"
            )
            base_result[text_key] = ""
            base_result[status_key] = "completed"
        return base_result

    semaphore = asyncio.Semaphore(max(1, int(cfg.transcribe_segment_concurrency)))
    done_count = 0

    async def infer(wav_b64: str, pcm: np.ndarray) -> dict:
        return await run_oneshot_asr(
            wav_b64,
            cfg=cfg,
            hotwords=hotwords,
            language=language,
            audio_pcm=pcm,
            hotword_pool_id=hotword_pool_id,
        )

    async def run_one(seg: OfflineSegment) -> dict:
        nonlocal done_count
        async with semaphore:
            entry: dict = {
                "id": seg.index,
                "start_ms": round(seg.start_ms),
                "end_ms": round(seg.end_ms),
                "text": "",
            }
            wav_b64 = pcm_to_wav_base64(seg.pcm)
            try:
                try:
                    res = await infer(wav_b64, seg.pcm)
                except Exception:
                    # One retry absorbs transient upstream hiccups; a second
                    # failure is recorded on the entry, not raised, so one bad
                    # segment can't sink the rest of the meeting.
                    res = await infer(wav_b64, seg.pcm)
                entry["text"] = str(res.get("text") or "")
                if res.get("language"):
                    entry["language"] = res["language"]
                should_refine = bool(
                    entry["text"]
                    and enhancement_options is not None
                    and (
                        enhancement_options.translate_mode
                        or enhancement_options.cleanup_level != "off"
                    )
                )
                if should_refine:
                    emotion = None
                    if enhancement_options.text_emotion:
                        try:
                            emotion = await infer_emotion(
                                float_pcm_to_i16_bytes(seg.pcm), enhancement_options
                            )
                            entry["_emotion_used"] = True
                        except Exception as exc:  # noqa: BLE001 - optional enhancement
                            logger.warning(
                                "Transcription segment %d emotion inference failed: %s",
                                seg.index,
                                exc,
                            )
                    try:
                        processed = await refine_text(
                            entry["text"], enhancement_options, emotion
                        )
                        if not enhancement_options.translate_mode:
                            accepted, reason = evaluate_cleanup_result(
                                entry["text"],
                                processed,
                                enhancement_options.cleanup_level,
                                enhancement_options.glossary,
                                preserve_punctuation=bool(emotion),
                            )
                            if not accepted:
                                raise ValueError(f"cleanup rejected: {reason}")
                        entry["_processed_text"] = processed
                    except Exception as exc:  # noqa: BLE001 - raw ASR remains usable
                        logger.warning(
                            "Transcription segment %d postprocess failed: %s",
                            seg.index,
                            exc,
                        )
                        entry["_postprocess_failed"] = True
            except Exception as exc:  # noqa: BLE001 - per-segment isolation
                logger.warning(
                    "Transcription segment %d failed after retry: %s",
                    seg.index,
                    exc,
                )
                entry["error"] = str(exc)
            done_count += 1
            if on_segment_done is not None:
                on_segment_done(done_count)
            return entry

    entries = await asyncio.gather(*(run_one(seg) for seg in segments))

    failed = [e for e in entries if "error" in e]
    if len(failed) == len(entries):
        raise JobExecutionError(
            "all segments failed ASR inference", code="inference_failed"
        )

    # Keep voiced segments and failed placeholders (so gaps are visible);
    # drop empty-text successes (noise that slipped past VAD).
    kept = [e for e in entries if e.get("text") or "error" in e]
    detected_lang = language or next(
        (e["language"] for e in entries if e.get("language")), ""
    )

    result = dict(base_result)
    result["language"] = detected_lang
    result["segments"] = kept
    result["full_text"] = "\n".join(e["text"] for e in kept if e.get("text"))
    result["failed_segments"] = len(failed)
    if enhancement_options is not None:
        result["text"] = result["full_text"]
        if enhancement_options.text_emotion:
            result["emotion_bucket_count"] = sum(
                1 for entry in kept if entry.pop("_emotion_used", False)
            )
        should_refine = (
            enhancement_options.translate_mode
            or enhancement_options.cleanup_level != "off"
        )
        if should_refine:
            status_key = (
                "translation_status"
                if enhancement_options.translate_mode
                else "cleanup_status"
            )
            text_key = (
                "translated_text"
                if enhancement_options.translate_mode
                else "cleaned_text"
            )
            postprocess_failed = False
            processed_parts: list[str] = []
            for entry in kept:
                postprocess_failed = bool(
                    entry.pop("_postprocess_failed", False)
                ) or postprocess_failed
                processed = entry.pop("_processed_text", "")
                if entry.get("text"):
                    processed_parts.append(processed)
            if postprocess_failed or len(processed_parts) != sum(
                1 for entry in kept if entry.get("text")
            ):
                result[status_key] = "degraded_raw_only"
            else:
                result[text_key] = "\n".join(processed_parts)
                result[status_key] = "completed"
    return result
