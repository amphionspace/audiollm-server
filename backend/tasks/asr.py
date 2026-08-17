"""ASR task engine: dual-model inference + fusion + pseudo-streaming partials."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..asr.client import query_audio_model_sync, query_audio_model_secondary_sync
from ..asr.ctc_streaming import CtcStreamingSession, config_from_app as ctc_config_from_app
from ..asr.fusion import choose_fused_result
from ..asr.itn import normalize_final_text
from ..asr.sherpa_streaming import SherpaStreamingSession, config_from_app
from ..audio.utils import pcm_to_wav_base64
from ..config import SAMPLE_RATE
from ..streaming.backpressure import (
    note_final_finished,
    note_final_queued,
    note_final_started,
)
from ..streaming.events import PartialSnapshot, SegmentReady
from ..streaming.session import SessionContext
from .base import BaseTaskEngine

logger = logging.getLogger(__name__)

_SEMAPHORES: dict[str, tuple[int, asyncio.Semaphore]] = {}
_FIRST_PARTIAL_COND = asyncio.Condition()
_PENDING_FIRST_PARTIALS = 0
_ACTIVE_FIRST_PARTIALS = 0
_FIRST_PARTIAL_PRIMARY_LOCK = asyncio.Lock()
_FIRST_PARTIAL_PRIMARY_INFLIGHT: dict[str, asyncio.Future] = {}
_PARTIAL_VLLM_EXECUTOR = ThreadPoolExecutor(
    max_workers=64, thread_name_prefix="asr-partial-vllm"
)
_EXECUTORS: dict[str, tuple[int, ThreadPoolExecutor]] = {}


def _get_semaphore(name: str, max_concurrent: int) -> asyncio.Semaphore:
    limit = max(1, int(max_concurrent))
    current = _SEMAPHORES.get(name)
    if current is None or current[0] != limit:
        sem = asyncio.Semaphore(limit)
        _SEMAPHORES[name] = (limit, sem)
        return sem
    return current[1]


def _get_executor(
    name: str, max_workers: int, *, thread_name_prefix: str
) -> ThreadPoolExecutor:
    workers = max(1, int(max_workers))
    current = _EXECUTORS.get(name)
    if current is not None and current[0] == workers:
        return current[1]
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=thread_name_prefix,
    )
    _EXECUTORS[name] = (workers, executor)
    if current is not None:
        current[1].shutdown(wait=False, cancel_futures=False)
    return executor


async def _try_acquire(
    sem: asyncio.Semaphore, timeout_ms: int | None
) -> bool:
    if timeout_ms is None:
        await sem.acquire()
        return True
    if timeout_ms <= 0:
        if sem.locked():
            return False
        await sem.acquire()
        return True
    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout_ms / 1000.0)
        return True
    except asyncio.TimeoutError:
        return False


async def _queue_first_partial_lane() -> None:
    global _PENDING_FIRST_PARTIALS
    async with _FIRST_PARTIAL_COND:
        _PENDING_FIRST_PARTIALS += 1
        _FIRST_PARTIAL_COND.notify_all()


async def _enter_first_partial_lane() -> None:
    global _ACTIVE_FIRST_PARTIALS, _PENDING_FIRST_PARTIALS
    async with _FIRST_PARTIAL_COND:
        _PENDING_FIRST_PARTIALS = max(0, _PENDING_FIRST_PARTIALS - 1)
        _ACTIVE_FIRST_PARTIALS += 1
        _FIRST_PARTIAL_COND.notify_all()


async def _cancel_queued_first_partial_lane() -> None:
    global _PENDING_FIRST_PARTIALS
    async with _FIRST_PARTIAL_COND:
        _PENDING_FIRST_PARTIALS = max(0, _PENDING_FIRST_PARTIALS - 1)
        _FIRST_PARTIAL_COND.notify_all()


async def _leave_first_partial_lane() -> None:
    global _ACTIVE_FIRST_PARTIALS
    async with _FIRST_PARTIAL_COND:
        _ACTIVE_FIRST_PARTIALS = max(0, _ACTIVE_FIRST_PARTIALS - 1)
        _FIRST_PARTIAL_COND.notify_all()


async def _wait_for_first_partial_lane(timeout_ms: int) -> float:
    if timeout_ms <= 0:
        return 0.0
    start = time.monotonic()
    deadline = start + timeout_ms / 1000.0
    async with _FIRST_PARTIAL_COND:
        while (_PENDING_FIRST_PARTIALS + _ACTIVE_FIRST_PARTIALS) > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(_FIRST_PARTIAL_COND.wait(), remaining)
            except asyncio.TimeoutError:
                break
    return time.monotonic() - start


async def _run_vllm_call(
    executor: ThreadPoolExecutor,
    func: Callable,
    /,
    *args,
    **kwargs,
):
    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, call)


def _split_primary_key(
    wav_b64: str,
    hotwords: list[str],
    ctx: SessionContext,
    request_kind: str,
) -> str:
    payload = {
        "request_kind": request_kind,
        "audio": wav_b64,
        "hotwords": hotwords,
        "src_lang": ctx.src_lang,
        "enrollment": ctx.enrollment_b64,
        "base_url": ctx.cfg.vllm_base_url,
        "model": ctx.cfg.vllm_model_name,
        "template": ctx.cfg.vllm_prompt_template,
        "repetition_penalty": ctx.cfg.asr_repetition_penalty,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AsrTaskEngine(BaseTaskEngine):
    """Drives the existing dual-ASR pipeline against the streaming session."""

    name = "asr"

    def __init__(self, *, emit_timing: bool = False) -> None:
        # When True, ``final`` messages carry the segment's session-timeline
        # position as ``bg_ms`` / ``ed_ms``. The native ``/transcribe-streaming``
        # contract is plain ``{type,text,language}``, so this stays off by
        # default and is only enabled for protocols that surface segment timing
        # (AST v3's ``bg`` / ``ed``). The wire protocol consumes these internal
        # fields; they never reach a native-framed client.
        self._emit_timing = emit_timing
        self._sherpa_session: SherpaStreamingSession | None = None
        self._sherpa_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Final segment -> final_asr / final
    # ------------------------------------------------------------------

    async def handle_segment(
        self, seg: SegmentReady, ctx: SessionContext
    ) -> bool:
        cfg = ctx.cfg
        segment = seg.pcm
        audio_duration = len(segment) / SAMPLE_RATE
        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(segment)
        hw_snapshot = ctx.hotwords
        tail_lane = bool(seg.is_stop_flush)
        if tail_lane:
            final_limit = int(
                getattr(
                    cfg,
                    "asr_stop_flush_max_concurrent",
                    getattr(cfg, "asr_final_max_concurrent", 8),
                )
            )
            final_executor_workers = int(
                getattr(cfg, "asr_stop_flush_executor_workers", final_limit)
            )
            final_lane = "final_stop_flush"
            final_thread_prefix = "asr-stop-flush-vllm"
        else:
            final_limit = int(getattr(cfg, "asr_final_max_concurrent", 8))
            final_executor_workers = int(
                getattr(cfg, "asr_final_executor_workers", final_limit)
            )
            final_lane = "final"
            final_thread_prefix = "asr-final-vllm"
        final_sem = _get_semaphore(final_lane, final_limit)
        final_executor = _get_executor(
            final_lane,
            final_executor_workers,
            thread_name_prefix=final_thread_prefix,
        )
        coalesce_future: asyncio.Future | None = None
        coalesce_owner = False
        coalesce_wait_elapsed = 0.0
        coalesce_enabled = (
            bool(getattr(cfg, "split_asr_enabled", False))
            and bool(cfg.enable_primary_asr)
            and not bool(cfg.enable_dual_asr_fusion)
            and not bool(cfg.enable_secondary_asr)
        )
        if coalesce_enabled:
            coalesce_key = _split_primary_key(wav_b64, hw_snapshot, ctx, "final")
            loop = asyncio.get_running_loop()
            async with _FIRST_PARTIAL_PRIMARY_LOCK:
                coalesce_future = _FIRST_PARTIAL_PRIMARY_INFLIGHT.get(coalesce_key)
                if coalesce_future is None:
                    coalesce_future = loop.create_future()
                    _FIRST_PARTIAL_PRIMARY_INFLIGHT[coalesce_key] = coalesce_future
                    coalesce_owner = True
        queue_start = time.monotonic()
        if coalesce_future is None or coalesce_owner:
            note_final_queued(final_limit)
        acquired_final = False
        if tail_lane:
            priority_wait_elapsed = 0.0
        else:
            priority_wait_elapsed = await _wait_for_first_partial_lane(
                int(getattr(cfg, "asr_first_partial_priority_ms", 0))
            )
        try:
            if coalesce_future is not None and not coalesce_owner:
                queue_elapsed = 0.0
            else:
                await final_sem.acquire()
                acquired_final = True
                note_final_started(final_limit)
                queue_elapsed = time.monotonic() - queue_start
            vllm_start = time.monotonic()

            primary_res: object = None
            secondary_res: object = None

            # Final segments only run the dual pipeline when fusion is on.
            # With fusion off (but secondary still online for partial gating)
            # we save one vLLM call per segment by running primary-only.
            if coalesce_future is not None and not coalesce_owner:
                t_coalesce = time.monotonic()
                primary_res = await coalesce_future
                coalesce_wait_elapsed = time.monotonic() - t_coalesce
            elif cfg.enable_dual_asr_fusion:
                secondary_res, primary_res = await self._dual_asr(
                    wav_b64,
                    hw_snapshot,
                    ctx,
                    request_kind="final",
                    executor=final_executor,
                    audio_pcm=segment,
                )
                if secondary_res is None and primary_res is None:
                    return False
            elif cfg.enable_primary_asr:
                primary_res = await asyncio.wait_for(
                    _run_vllm_call(
                        final_executor,
                        query_audio_model_sync,
                        wav_b64,
                        hotwords=hw_snapshot,
                        src_lang=ctx.src_lang,
                        audio_pcm=segment,
                        audio_sample_rate=SAMPLE_RATE,
                        enrollment_wav_base64=ctx.enrollment_b64,
                        enrollment_id=ctx.enrollment_id,
                        hotword_pool_id=getattr(ctx, "hotword_pool_id", ""),
                        base_url=cfg.vllm_base_url,
                        model_name=cfg.vllm_model_name,
                        prompt_template=cfg.vllm_prompt_template,
                        timeout=cfg.asr_request_timeout,
                        repetition_penalty=cfg.asr_repetition_penalty,
                        trace_id=ctx.trace_id or "",
                        request_kind="final",
                    ),
                    timeout=cfg.primary_asr_timeout,
                )
            if coalesce_owner and coalesce_future is not None and not coalesce_future.done():
                coalesce_future.set_result(primary_res)
        except BaseException as exc:
            if coalesce_owner and coalesce_future is not None and not coalesce_future.done():
                coalesce_future.set_exception(exc)
            raise
        finally:
            if acquired_final:
                final_sem.release()
                note_final_finished()
            if coalesce_owner and coalesce_future is not None:
                async with _FIRST_PARTIAL_PRIMARY_LOCK:
                    for key, future in list(_FIRST_PARTIAL_PRIMARY_INFLIGHT.items()):
                        if future is coalesce_future:
                            _FIRST_PARTIAL_PRIMARY_INFLIGHT.pop(key, None)
                            break

        primary_result = (
            None if isinstance(primary_res, Exception) else primary_res
        )
        secondary_result = (
            None if isinstance(secondary_res, Exception) else secondary_res
        )

        if isinstance(primary_res, Exception):
            logger.warning("Primary ASR failed: %s", primary_res)
        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed: %s", secondary_res)
        if primary_result is None and secondary_result is None:
            raise RuntimeError("Both ASR models failed for this segment.")

        text, detected_lang = self._select_text(
            primary_result, secondary_result, hw_snapshot, ctx
        )
        if text:
            text = normalize_final_text(text, detected_lang, cfg)

        elapsed = time.monotonic() - t0
        before_send_at = time.monotonic()
        api_infer_elapsed = before_send_at - vllm_start
        session_queue_elapsed = (
            max(0.0, seg.dequeued_at - seg.queued_at)
            if seg.queued_at > 0 and seg.dequeued_at > 0
            else 0.0
        )
        segment_age_before_send = (
            max(0.0, before_send_at - seg.queued_at) if seg.queued_at > 0 else elapsed
        )
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        api_infer_rtf = api_infer_elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Final ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration, elapsed, rtf, text[:80],
        )
        if not text:
            logger.info(
                "ASR_TIMING type=final traceId=%s audio_sec=%.3f "
                "session_queue_ms=%.1f queue_ms=%.1f final_queue_ms=%.1f "
                "priority_wait_ms=%.1f api_infer_ms=%.1f "
                "send_ms=0.0 segment_total_ms=%.1f api_infer_rtf=%.4f "
                "queue_depth_at_enqueue=%s stop_flush=%s tail_lane=%s "
                "coalesce_owner=%s coalesce_wait_ms=%.1f "
                "start_ms=%s end_ms=%s text=%r",
                ctx.trace_id or "-",
                audio_duration,
                session_queue_elapsed * 1000.0,
                queue_elapsed * 1000.0,
                queue_elapsed * 1000.0,
                priority_wait_elapsed * 1000.0,
                api_infer_elapsed * 1000.0,
                segment_age_before_send * 1000.0,
                api_infer_rtf,
                seg.queue_depth_at_enqueue,
                seg.is_stop_flush,
                tail_lane,
                coalesce_owner,
                coalesce_wait_elapsed * 1000.0,
                seg.start_ms,
                seg.end_ms,
                text[:80],
            )
            await self._close_sherpa_partial(ctx)
            return False

        payload: dict = {
            "type": "final",
            "text": text,
            "language": detected_lang,
        }
        if self._emit_timing:
            if seg.start_ms is not None:
                payload["bg_ms"] = seg.start_ms
            if seg.end_ms is not None:
                payload["ed_ms"] = seg.end_ms
        sent = await ctx.send_json(payload)
        sent_at = time.monotonic()
        send_elapsed = sent_at - before_send_at
        segment_total_elapsed = (
            max(0.0, sent_at - seg.queued_at) if seg.queued_at > 0 else elapsed
        )
        logger.info(
            "ASR_TIMING type=final traceId=%s audio_sec=%.3f "
            "session_queue_ms=%.1f queue_ms=%.1f final_queue_ms=%.1f "
            "priority_wait_ms=%.1f api_infer_ms=%.1f send_ms=%.1f "
            "segment_total_ms=%.1f api_infer_rtf=%.4f "
            "queue_depth_at_enqueue=%s stop_flush=%s tail_lane=%s "
            "coalesce_owner=%s coalesce_wait_ms=%.1f "
            "start_ms=%s end_ms=%s text=%r",
            ctx.trace_id or "-",
            audio_duration,
            session_queue_elapsed * 1000.0,
            queue_elapsed * 1000.0,
            queue_elapsed * 1000.0,
            priority_wait_elapsed * 1000.0,
            api_infer_elapsed * 1000.0,
            send_elapsed * 1000.0,
            segment_total_elapsed * 1000.0,
            api_infer_rtf,
            seg.queue_depth_at_enqueue,
            seg.is_stop_flush,
            tail_lane,
            coalesce_owner,
            coalesce_wait_elapsed * 1000.0,
            seg.start_ms,
            seg.end_ms,
            text[:80],
        )
        await self._close_sherpa_partial(ctx)
        return sent

    # ------------------------------------------------------------------
    # Pseudo-streaming partial
    # ------------------------------------------------------------------

    async def handle_partial(
        self, snap: PartialSnapshot, ctx: SessionContext
    ) -> None:
        cfg = ctx.cfg
        if not cfg.enable_pseudo_stream:
            return
        if self._use_sherpa_partial(cfg):
            await self._handle_sherpa_partial(snap, ctx)
            return
        if not (cfg.enable_primary_asr or cfg.enable_secondary_asr):
            return

        snapshot = snap.pcm
        audio_duration = len(snapshot) / SAMPLE_RATE
        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(snapshot)
        pcm_encode_elapsed = time.monotonic() - t0
        hw_snapshot = ctx.hotwords
        coalesce_future: asyncio.Future | None = None
        coalesce_owner = False
        coalesce_wait_elapsed = 0.0
        coalesce_enabled = (
            snap.is_first
            and bool(getattr(cfg, "split_asr_enabled", False))
            and bool(cfg.enable_primary_asr)
            and not bool(cfg.enable_secondary_asr)
        )
        if coalesce_enabled:
            coalesce_key = _split_primary_key(
                wav_b64, hw_snapshot, ctx, "partial_first"
            )
            loop = asyncio.get_running_loop()
            async with _FIRST_PARTIAL_PRIMARY_LOCK:
                coalesce_future = _FIRST_PARTIAL_PRIMARY_INFLIGHT.get(coalesce_key)
                if coalesce_future is None:
                    coalesce_future = loop.create_future()
                    _FIRST_PARTIAL_PRIMARY_INFLIGHT[coalesce_key] = coalesce_future
                    coalesce_owner = True
        if snap.is_first:
            partial_sem = _get_semaphore(
                "partial_first",
                getattr(cfg, "asr_first_partial_max_concurrent", 8),
            )
            queue_timeout_ms: int | None = None
        else:
            refresh_limit = getattr(cfg, "asr_partial_refresh_max_concurrent", 0)
            if int(refresh_limit) <= 0:
                logger.info(
                    "ASR_TIMING type=partial_skipped traceId=%s first=%s "
                    "audio_sec=%.3f queue_ms=0.0 reason=partial_refresh_disabled",
                    ctx.trace_id or "-",
                    snap.is_first,
                    audio_duration,
                )
                return
            partial_sem = _get_semaphore(
                "partial_refresh",
                refresh_limit,
            )
            queue_timeout_ms = getattr(cfg, "asr_partial_refresh_queue_timeout_ms", 50)

        queue_start = time.monotonic()
        acquired_sem = False
        if coalesce_future is not None and not coalesce_owner:
            acquired = True
        else:
            acquired = await _try_acquire(partial_sem, queue_timeout_ms)
            acquired_sem = acquired
        queue_elapsed = time.monotonic() - queue_start
        if not acquired:
            logger.info(
                "ASR_TIMING type=partial_skipped traceId=%s first=%s "
                "audio_sec=%.3f queue_ms=%.1f reason=partial_lane_saturated",
                ctx.trace_id or "-",
                snap.is_first,
                audio_duration,
                queue_elapsed * 1000.0,
            )
            return
        snap.model_started_at = time.monotonic()
        vllm_start = snap.model_started_at
        speech_started_at = snap.speech_started_at or ctx.last_speech_started_at
        speech_to_snapshot_ms = (
            ((snap.snapshot_at or vllm_start) - speech_started_at) * 1000.0
            if speech_started_at is not None
            else -1.0
        )
        snapshot_to_lane_ms = (
            (vllm_start - snap.snapshot_at) * 1000.0 if snap.snapshot_at else -1.0
        )

        primary_res: object = None
        secondary_res: object = None
        lane_entered = False
        first_pending = False

        try:
            if snap.is_first:
                await _queue_first_partial_lane()
                first_pending = True
                await _enter_first_partial_lane()
                first_pending = False
                lane_entered = True
            if coalesce_future is not None and not coalesce_owner:
                t_coalesce = time.monotonic()
                primary_res = await coalesce_future
                coalesce_wait_elapsed = time.monotonic() - t_coalesce
            elif cfg.enable_secondary_asr and cfg.enable_primary_asr:
                secondary_res, primary_res = await self._dual_asr(
                    wav_b64,
                    hw_snapshot,
                    ctx,
                    request_kind="partial_first" if snap.is_first else "partial_refresh",
                    executor=_PARTIAL_VLLM_EXECUTOR,
                )
                if secondary_res is None and primary_res is None:
                    return
            elif cfg.enable_primary_asr:
                primary_res = await asyncio.wait_for(
                    _run_vllm_call(
                        _PARTIAL_VLLM_EXECUTOR,
                        query_audio_model_sync,
                        wav_b64,
                        hotwords=hw_snapshot,
                        src_lang=ctx.src_lang,
                        enrollment_wav_base64=ctx.enrollment_b64,
                        enrollment_id=ctx.enrollment_id,
                        base_url=cfg.vllm_base_url,
                        model_name=cfg.vllm_model_name,
                        prompt_template=cfg.vllm_prompt_template,
                        timeout=cfg.asr_request_timeout,
                        repetition_penalty=cfg.asr_repetition_penalty,
                        trace_id=ctx.trace_id or "",
                        request_kind=(
                            "partial_first" if snap.is_first else "partial_refresh"
                        ),
                    ),
                    timeout=cfg.primary_asr_timeout,
                )
            elif cfg.enable_secondary_asr:
                secondary_res = await _run_vllm_call(
                    _PARTIAL_VLLM_EXECUTOR,
                    query_audio_model_secondary_sync,
                    wav_b64,
                    hotwords=hw_snapshot,
                    base_url=cfg.secondary_vllm_base_url,
                    model_name=cfg.secondary_vllm_model_name,
                    timeout=cfg.asr_request_timeout,
                    trace_id=ctx.trace_id or "",
                    request_kind=(
                        "partial_first_secondary"
                        if snap.is_first
                        else "partial_refresh_secondary"
                    ),
                )
            if coalesce_owner and coalesce_future is not None and not coalesce_future.done():
                coalesce_future.set_result(primary_res)
        except BaseException as exc:
            if coalesce_owner and coalesce_future is not None and not coalesce_future.done():
                coalesce_future.set_exception(exc)
            raise
        finally:
            if first_pending:
                await _cancel_queued_first_partial_lane()
            if lane_entered:
                await _leave_first_partial_lane()
            if acquired_sem:
                partial_sem.release()
            if coalesce_owner and coalesce_future is not None:
                async with _FIRST_PARTIAL_PRIMARY_LOCK:
                    for key, future in list(_FIRST_PARTIAL_PRIMARY_INFLIGHT.items()):
                        if future is coalesce_future:
                            _FIRST_PARTIAL_PRIMARY_INFLIGHT.pop(key, None)
                            break

        primary_result = (
            None if isinstance(primary_res, Exception) else primary_res
        )
        secondary_result = (
            None if isinstance(secondary_res, Exception) else secondary_res
        )

        if primary_result is None and secondary_result is None:
            return

        # Noise gate: if secondary is enabled and produced empty text, skip.
        if cfg.enable_secondary_asr:
            sec_text = str(
                (secondary_result or {}).get("transcription") or ""
            ).strip()
            if not sec_text:
                logger.debug("Partial suppressed: secondary empty (noise gate)")
                return

        text, _ = self._select_text(
            primary_result, secondary_result, hw_snapshot, ctx
        )

        partial_ready_at = time.monotonic()
        elapsed = partial_ready_at - t0
        api_infer_elapsed = partial_ready_at - vllm_start
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        api_infer_rtf = api_infer_elapsed / audio_duration if audio_duration > 0 else 0.0
        speech_to_partial_ms = (
            (partial_ready_at - speech_started_at) * 1000.0
            if speech_started_at is not None
            else -1.0
        )
        snapshot_to_partial_ms = (partial_ready_at - vllm_start) * 1000.0
        logger.info(
            "Partial ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration, elapsed, rtf, text[:80],
        )
        logger.info(
            "ASR_TIMING type=partial traceId=%s audio_sec=%.3f "
            "first=%s queue_ms=%.1f speech_to_partial_ms=%.1f "
            "snapshot_to_partial_ms=%.1f api_infer_ms=%.1f "
            "api_infer_rtf=%.4f speech_to_snapshot_ms=%.1f "
            "snapshot_to_lane_ms=%.1f pcm_encode_ms=%.1f "
            "coalesce_owner=%s coalesce_wait_ms=%.1f text=%r",
            ctx.trace_id or "-",
            audio_duration,
            snap.is_first,
            queue_elapsed * 1000.0,
            speech_to_partial_ms,
            snapshot_to_partial_ms,
            api_infer_elapsed * 1000.0,
            api_infer_rtf,
            speech_to_snapshot_ms,
            snapshot_to_lane_ms,
            pcm_encode_elapsed * 1000.0,
            coalesce_owner,
            coalesce_wait_elapsed * 1000.0,
            text[:80],
        )

        if not text:
            return

        send_start = time.monotonic()
        sent = await ctx.send_json(
            {
                "type": "partial",
                "text": text,
                "language": ctx.language,
            }
        )
        send_done = time.monotonic()
        logger.info(
            "ASR_TIMING type=partial_send traceId=%s first=%s "
            "send_ms=%.1f speech_to_send_ms=%.1f sent=%s text_chars=%d",
            ctx.trace_id or "-",
            snap.is_first,
            (send_done - send_start) * 1000.0,
            (
                (send_done - speech_started_at) * 1000.0
                if speech_started_at is not None
                else -1.0
            ),
            sent,
            len(text),
        )

    async def handle_speech_start(self, ctx: SessionContext) -> None:
        await self._reset_sherpa_partial(ctx)
        # Notify the client that VAD just activated.  Clients use this
        # timestamp as the reference for TTFT measurement so silence at
        # the start of the clip is excluded from the reported latency.
        await ctx.send_json({"type": "speech_started"})

    async def handle_speech_dropped(self, ctx: SessionContext) -> None:
        await self._close_sherpa_partial(ctx)

    # ------------------------------------------------------------------
    # Stop guarantee: always emit a final after stop (possibly empty).
    # ------------------------------------------------------------------

    async def on_stop(
        self,
        ctx: SessionContext,
        *,
        sent_any_response: bool,
        stopped: bool,
    ) -> None:
        await self._close_sherpa_partial(ctx)
        # Match the legacy behavior: only emit empty final after explicit stop
        # (not after raw socket close) when nothing was sent in this drain.
        if stopped and not sent_any_response:
            await ctx.send_json(
                {
                    "type": "final",
                    "text": "",
                    "language": ctx.language,
                }
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _use_sherpa_partial(self, cfg) -> bool:
        backend = str(getattr(cfg, "streaming_partial_backend", "vllm") or "vllm")
        return backend.strip().lower() in {"sherpa", "k2_om", "ctc_om"}

    def _get_sherpa_session_sync(
        self, ctx: SessionContext
    ) -> SherpaStreamingSession | CtcStreamingSession:
        session = ctx.runtime_state.get("sherpa_session")
        if session is None:
            backend = str(
                getattr(ctx.cfg, "streaming_partial_backend", "vllm") or "vllm"
            ).strip().lower()
            if backend == "ctc_om":
                session = CtcStreamingSession(ctc_config_from_app(ctx.cfg))
            else:
                session = SherpaStreamingSession(
                    config_from_app(ctx.cfg, session_key=ctx.trace_id)
                )
            ctx.runtime_state["sherpa_session"] = session
        return session

    def _accept_sherpa_cumulative_sync(
        self, ctx: SessionContext, snapshot, max_decode_steps: int = 0
    ) -> tuple[str, dict]:
        session = self._get_sherpa_session_sync(ctx)
        return session.accept_cumulative(
            snapshot,
            max_decode_steps=int(max_decode_steps),
            trace_id=ctx.trace_id or "",
        )

    def _sherpa_lock_for(self, ctx: SessionContext) -> asyncio.Lock:
        lock = ctx.runtime_state.get("sherpa_lock")
        if lock is None:
            lock = asyncio.Lock()
            ctx.runtime_state["sherpa_lock"] = lock
        return lock

    async def prefeed_streaming_partial(
        self, snap: PartialSnapshot, ctx: SessionContext
    ) -> None:
        backend = str(
            getattr(ctx.cfg, "streaming_partial_backend", "vllm") or "vllm"
        ).strip().lower()
        if backend != "ctc_om":
            return
        executor = _get_executor(
            "partial_sherpa",
            int(getattr(ctx.cfg, "sherpa_executor_workers", 4)),
            thread_name_prefix="asr-partial-sherpa",
        )
        async with self._sherpa_lock_for(ctx):
            text, stats = await _run_vllm_call(
                executor,
                self._accept_sherpa_cumulative_sync,
                ctx,
                snap.pcm,
                -1,
            )
        if not bool(getattr(ctx.cfg, "ctc_prefeed_emit_enabled", False)):
            return
        text = str(text or "").strip()
        if not text:
            return
        if ctx.runtime_state.get("ctc_prefeed_last_text") == text:
            return
        now = time.monotonic()
        interval_ms = max(
            0,
            int(getattr(ctx.cfg, "ctc_partial_post_first_interval_ms", 1500) or 0),
        )
        last_emit_at = float(ctx.runtime_state.get("ctc_partial_last_emit_at") or 0.0)
        if (
            last_emit_at > 0
            and interval_ms > 0
            and (now - last_emit_at) * 1000.0 < interval_ms
        ):
            return
        ctx.runtime_state["ctc_partial_last_emit_at"] = now
        ctx.runtime_state["ctc_prefeed_last_text"] = text
        sent = await ctx.send_json(
            {
                "type": "partial",
                "text": text,
                "language": ctx.language,
            }
        )
        logger.info(
            "ASR_TIMING type=ctc_prefeed_send traceId=%s audio_sec=%.3f "
            "sent=%s text_chars=%s batch_size=%s decode_ms=%.1f total_ms=%.1f",
            ctx.trace_id or "-",
            len(snap.pcm) / SAMPLE_RATE,
            sent,
            len(text),
            int(stats.get("batch_size", 0)),
            float(stats.get("decode_ms", 0.0)),
            float(stats.get("total_ms", 0.0)),
        )

    async def poll_cached_streaming_partial(self, ctx: SessionContext) -> bool:
        backend = str(
            getattr(ctx.cfg, "streaming_partial_backend", "vllm") or "vllm"
        ).strip().lower()
        if backend != "ctc_om":
            return False
        session = ctx.runtime_state.get("sherpa_session")
        if session is None or not hasattr(session, "cached_result"):
            return False
        text, stats = session.cached_result()
        text = str(text or "").strip()
        if not text or ctx.runtime_state.get("ctc_prefeed_last_text") == text:
            return False
        now = time.monotonic()
        interval_ms = max(
            0,
            int(getattr(ctx.cfg, "ctc_partial_post_first_interval_ms", 1500) or 0),
        )
        last_emit_at = float(ctx.runtime_state.get("ctc_partial_last_emit_at") or 0.0)
        if (
            last_emit_at > 0
            and interval_ms > 0
            and (now - last_emit_at) * 1000.0 < interval_ms
        ):
            return False
        ctx.runtime_state["ctc_partial_last_emit_at"] = now
        ctx.runtime_state["ctc_prefeed_last_text"] = text
        sent = await ctx.send_json(
            {
                "type": "partial",
                "text": text,
                "language": ctx.language,
            }
        )
        logger.info(
            "ASR_TIMING type=ctc_cached_send traceId=%s sent=%s text_chars=%s "
            "batch_size=%s decode_ms=%.1f total_ms=%.1f",
            ctx.trace_id or "-",
            sent,
            len(text),
            int(stats.get("batch_size", 0)),
            float(stats.get("decode_ms", 0.0)),
            float(stats.get("total_ms", 0.0)),
        )
        return bool(sent)

    def install_streaming_result_callback(self, ctx: SessionContext, callback) -> bool:
        backend = str(
            getattr(ctx.cfg, "streaming_partial_backend", "vllm") or "vllm"
        ).strip().lower()
        if backend != "ctc_om":
            return False
        session = ctx.runtime_state.get("sherpa_session")
        if session is None:
            session = self._get_sherpa_session_sync(ctx)
        if session is None or not hasattr(session, "set_result_callback"):
            return False
        session.set_result_callback(callback)
        return True

    async def close_streaming_partial(self, ctx: SessionContext) -> None:
        await self._close_sherpa_partial(ctx)

    async def _reset_sherpa_partial(self, ctx: SessionContext) -> None:
        backend = str(
            getattr(ctx.cfg, "streaming_partial_backend", "vllm") or "vllm"
        ).strip().lower()
        if backend == "ctc_om":
            # CTC partial relies on the pretrigger audio already fed before VAD
            # speech_start. Resetting here drops the first 200 ms and breaks the
            # fixed BS52 slot scheduler before it reaches the first 77-frame tick.
            return
        async with self._sherpa_lock_for(ctx):
            session = ctx.runtime_state.get("sherpa_session")
            if session is not None:
                session.reset()

    async def _close_sherpa_partial(self, ctx: SessionContext) -> None:
        async with self._sherpa_lock_for(ctx):
            session = ctx.runtime_state.pop("sherpa_session", None)
            if session is not None:
                session.close()

    async def _handle_sherpa_partial(
        self, snap: PartialSnapshot, ctx: SessionContext
    ) -> None:
        cfg = ctx.cfg
        snapshot = snap.pcm
        audio_duration = len(snapshot) / SAMPLE_RATE
        start = time.monotonic()
        speech_started_at = snap.speech_started_at or ctx.last_speech_started_at
        speech_to_snapshot_ms = (
            ((snap.snapshot_at or start) - speech_started_at) * 1000.0
            if speech_started_at is not None
            else -1.0
        )
        executor = _get_executor(
            "partial_sherpa",
            int(getattr(cfg, "sherpa_executor_workers", 4)),
            thread_name_prefix="asr-partial-sherpa",
        )
        max_decode_steps = int(
            getattr(
                cfg,
                (
                    "k2_first_partial_decode_max_steps"
                    if snap.is_first
                    else "k2_partial_decode_max_steps"
                ),
                0,
            )
            or 0
        )
        snap.model_started_at = time.monotonic()
        async with self._sherpa_lock_for(ctx):
            decode_start = time.monotonic()
            text, k2_stats = await _run_vllm_call(
                executor,
                self._accept_sherpa_cumulative_sync,
                ctx,
                snapshot,
                max_decode_steps,
            )
        ready_at = time.monotonic()
        api_infer_elapsed = ready_at - decode_start
        elapsed = ready_at - start
        speech_to_partial_ms = (
            (ready_at - speech_started_at) * 1000.0
            if speech_started_at is not None
            else -1.0
        )
        logger.info(
            "ASR_TIMING type=partial_sherpa traceId=%s audio_sec=%.3f "
            "first=%s speech_to_partial_ms=%.1f snapshot_to_partial_ms=%.1f "
            "api_infer_ms=%.1f api_infer_rtf=%.4f "
            "speech_to_snapshot_ms=%.1f k2_queue_ms=%.1f "
            "k2_batch_wait_ms=%.1f k2_lock_wait_ms=%.1f "
            "k2_accept_ms=%.1f k2_decode_ms=%.1f k2_result_ms=%.1f "
            "k2_total_ms=%.1f k2_decode_loops=%s k2_batch_size=%s text=%r",
            ctx.trace_id or "-",
            audio_duration,
            snap.is_first,
            speech_to_partial_ms,
            (ready_at - start) * 1000.0,
            api_infer_elapsed * 1000.0,
            api_infer_elapsed / audio_duration if audio_duration > 0 else 0.0,
            speech_to_snapshot_ms,
            float(k2_stats.get("queue_ms", 0.0)),
            float(k2_stats.get("batch_wait_ms", 0.0)),
            float(k2_stats.get("lock_wait_ms", 0.0)),
            float(k2_stats.get("accept_ms", 0.0)),
            float(k2_stats.get("decode_ms", 0.0)),
            float(k2_stats.get("result_ms", 0.0)),
            float(k2_stats.get("total_ms", 0.0)),
            int(k2_stats.get("decode_loops", 0)),
            int(k2_stats.get("batch_size", 0)),
            text[:80],
        )
        if not text:
            return
        send_start = time.monotonic()
        sent = await ctx.send_json(
            {
                "type": "partial",
                "text": text,
                "language": ctx.language,
            }
        )
        send_done = time.monotonic()
        logger.info(
            "ASR_TIMING type=partial_sherpa_send traceId=%s first=%s "
            "send_ms=%.1f speech_to_send_ms=%.1f sent=%s text_chars=%d "
            "total_ms=%.1f",
            ctx.trace_id or "-",
            snap.is_first,
            (send_done - send_start) * 1000.0,
            (
                (send_done - speech_started_at) * 1000.0
                if speech_started_at is not None
                else -1.0
            ),
            sent,
            len(text),
            elapsed * 1000.0,
        )

    async def _dual_asr(
        self,
        wav_b64: str,
        hw_snapshot: list[str],
        ctx: SessionContext,
        *,
        request_kind: str,
        executor: ThreadPoolExecutor,
        audio_pcm: np.ndarray | None = None,
    ) -> tuple:
        cfg = ctx.cfg
        secondary_task = asyncio.create_task(
            _run_vllm_call(
                executor,
                query_audio_model_secondary_sync,
                wav_b64,
                hotwords=hw_snapshot,
                base_url=cfg.secondary_vllm_base_url,
                model_name=cfg.secondary_vllm_model_name,
                timeout=cfg.asr_request_timeout,
                trace_id=ctx.trace_id or "",
                request_kind=f"{request_kind}_secondary",
            )
        )
        primary_task = None
        if cfg.enable_primary_asr:
            primary_task = asyncio.create_task(
                asyncio.wait_for(
                    _run_vllm_call(
                        executor,
                        query_audio_model_sync,
                        wav_b64,
                        hotwords=hw_snapshot,
                        src_lang=ctx.src_lang,
                        audio_pcm=audio_pcm,
                        audio_sample_rate=SAMPLE_RATE,
                        enrollment_wav_base64=ctx.enrollment_b64,
                        enrollment_id=ctx.enrollment_id,
                        hotword_pool_id=getattr(ctx, "hotword_pool_id", ""),
                        base_url=cfg.vllm_base_url,
                        model_name=cfg.vllm_model_name,
                        prompt_template=cfg.vllm_prompt_template,
                        timeout=cfg.asr_request_timeout,
                        repetition_penalty=cfg.asr_repetition_penalty,
                        trace_id=ctx.trace_id or "",
                        request_kind=f"{request_kind}_primary",
                    ),
                    timeout=cfg.primary_asr_timeout,
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

    def _select_text(
        self,
        primary_result,
        secondary_result,
        hw_snapshot: list[str],
        ctx: SessionContext,
    ) -> tuple[str, str]:
        cfg = ctx.cfg
        detected_lang = ctx.language

        if primary_result and not secondary_result:
            text = str(primary_result.get("transcription") or "").strip()
            detected_lang = (
                primary_result.get("detected_language") or ctx.language
            )
        elif secondary_result and not primary_result:
            text = str(secondary_result.get("transcription") or "").strip()
        else:
            fused = choose_fused_result(
                primary_result,
                secondary_result,
                hotwords=hw_snapshot,
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

        return text, detected_lang
