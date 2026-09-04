"""ASR task engine: dual-model inference + fusion + pseudo-streaming partials."""

from __future__ import annotations

import asyncio
import logging
import time

from ..asr.client import (
    diagnostic_hotword_misses,
    hotword_hits_in_text,
    query_audio_model,
    query_audio_model_secondary,
)
from ..asr.fusion import choose_fused_result
from ..asr.itn import normalize_final_text
from ..audio.utils import pcm_to_wav_base64
from ..audio.vad import filter_speech_for_asr, prepare_audio_for_asr
from ..config import SAMPLE_RATE
from ..diarization.turns import split_segment_by_speaker
from ..streaming.events import PartialSnapshot, SegmentReady
from ..streaming.session import SessionContext
from .base import BaseTaskEngine

logger = logging.getLogger(__name__)


def _public_language(value: object) -> str:
    """Return an outbound language value without placeholder labels."""
    language = str(value or "").strip()
    if language.lower() in {"n/a", "na", "unknown", "und", "none", "null", "-"}:
        return ""
    return language


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

    # ------------------------------------------------------------------
    # Final segment -> final_asr / final
    # ------------------------------------------------------------------

    async def handle_segment(self, seg: SegmentReady, ctx: SessionContext) -> bool:
        segments = [seg]
        if (
            ctx.cfg.enable_role_separation
            and ctx.diarization is not None
            and seg.start_ms is not None
            and seg.end_ms is not None
        ):
            turns = await ctx.diarization.turns_for(seg.start_ms, seg.end_ms)
            if turns:
                segments = split_segment_by_speaker(
                    seg,
                    turns,
                    min_duration_ms=ctx.cfg.min_segment_duration_ms,
                )
                logger.info(
                    "Diarization split: session_id=%s segment_id=%s turns=%d outputs=%d",
                    ctx.session_id,
                    seg.id or "n/a",
                    len(turns),
                    len(segments),
                )

        sent_any = False
        for attributed in segments:
            sent_any = await self._handle_single_segment(attributed, ctx) or sent_any
        return sent_any

    async def _handle_single_segment(self, seg: SegmentReady, ctx: SessionContext) -> bool:
        cfg = ctx.cfg
        segment = seg.pcm
        original_duration = len(segment) / SAMPLE_RATE
        prepared = prepare_audio_for_asr(segment, cfg)
        trim = prepared.silence_removal
        if trim.removed_ranges:
            logger.info(
                "Final ASR silence removal: segment_id=%s original=%.2fs "
                "removed=%.2fs ranges=%d trimmed=%.2fs threshold=%.2fs",
                seg.id or "n/a",
                original_duration,
                trim.removed_sec,
                trim.removed_ranges,
                len(trim.pcm) / SAMPLE_RATE,
                cfg.asr_silence_removal_threshold_sec,
            )
        voice_filter = prepared.speech_filter
        voice = voice_filter.evidence
        if not voice.accepted:
            logger.info(
                "Final ASR skipped by segment voice gate: reason=%s "
                "segment_id=%s audio=%.2fs speech_ms=%.0f ratio=%.3f "
                "frames=%d/%d max_prob=%.3f mean_prob=%.3f rms=%.5f",
                voice.reason,
                seg.id or "n/a",
                len(trim.pcm) / SAMPLE_RATE,
                voice.speech_ms,
                voice.speech_ratio,
                voice.speech_frames,
                voice.total_frames,
                voice.max_prob,
                voice.mean_prob,
                voice.rms,
            )
            return False
        if voice_filter.kept_samples < voice_filter.input_samples:
            logger.info(
                "Final ASR speech filter: segment_id=%s input=%.2fs kept=%.2fs "
                "ranges=%d speech_ratio=%.3f",
                seg.id or "n/a",
                voice_filter.input_samples / SAMPLE_RATE,
                voice_filter.kept_samples / SAMPLE_RATE,
                voice_filter.kept_ranges,
                voice.speech_ratio,
            )
        segment = prepared.pcm
        audio_duration = len(segment) / SAMPLE_RATE
        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(segment)
        hw_snapshot = ctx.hotwords

        primary_res: object = None
        secondary_res: object = None

        # Final segments only run the dual pipeline when fusion is on.
        # With fusion off (but secondary still online for partial gating)
        # we save one vLLM call per segment by running primary-only.
        if cfg.enable_dual_asr_fusion:
            secondary_res, primary_res = await self._dual_asr(
                wav_b64, hw_snapshot, ctx, audio_pcm=segment
            )
            if secondary_res is None and primary_res is None:
                return False
        elif cfg.enable_primary_asr:
            primary_res = await asyncio.wait_for(
                query_audio_model(
                    wav_b64,
                    hotwords=hw_snapshot,
                    src_lang=ctx.src_lang,
                    audio_pcm=segment,
                    audio_sample_rate=SAMPLE_RATE,
                    enrollment_wav_base64=ctx.enrollment_b64,
                    base_url=cfg.vllm_base_url,
                    model_name=cfg.vllm_model_name,
                    prompt_template=cfg.vllm_prompt_template,
                    timeout=cfg.asr_request_timeout,
                    runtime_config=cfg,
                    hotword_pool_id=ctx.hotword_pool_id,
                    enrollment_id=ctx.enrollment_id,
                    enrollment_scope_id=cfg.hotword_pool_id,
                    session_id=ctx.session_id,
                    gateway_trace_id=ctx.gateway_trace_id,
                ),
                timeout=cfg.primary_asr_timeout,
            )

        primary_result = None if isinstance(primary_res, Exception) else primary_res
        secondary_result = None if isinstance(secondary_res, Exception) else secondary_res

        if isinstance(primary_res, Exception):
            logger.warning("Primary ASR failed: %s", primary_res)
        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed: %s", secondary_res)
        if primary_result is None and secondary_result is None:
            raise RuntimeError("Both ASR models failed for this segment.")

        text, detected_lang = self._select_text(primary_result, secondary_result, hw_snapshot, ctx)
        # ITN + plate normalization is a final-only display transform; partials
        # stay spoken-form (see handle_partial).
        if text:
            text = normalize_final_text(text, detected_lang, cfg)
        effective_hotwords = self._effective_hotwords_for_final(
            primary_result,
            hw_snapshot,
        )
        returned_effective_hotwords = self._rag_recalled_hotwords_for_final(primary_result)
        if text and cfg.vllm_prompt_template == "qwen3_asr" and effective_hotwords:
            text = await self._refine_qwen_hotwords(
                text,
                language=ctx.language,
                hotwords=effective_hotwords,
            )
        hotword_hits = hotword_hits_in_text(effective_hotwords, text)

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Final ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration,
            elapsed,
            rtf,
            text[:80],
        )
        logger.info(
            "Final ASR diagnostic: session_id=%s gateway_trace_id=%s "
            "segment_id=%s final_text=%r effective_hotwords_count=%d "
            "effective_hotword_hits=%s",
            ctx.session_id or "n/a",
            ctx.gateway_trace_id or "n/a",
            seg.id or "n/a",
            text,
            len(effective_hotwords),
            hotword_hits,
        )
        hotword_miss = diagnostic_hotword_misses(effective_hotwords, text)
        if hotword_miss:
            logger.warning(
                "Final ASR hotword miss: session_id=%s gateway_trace_id=%s "
                "segment_id=%s hotword_miss=%s",
                ctx.session_id or "n/a",
                ctx.gateway_trace_id or "n/a",
                seg.id or "n/a",
                hotword_miss,
            )

        # Dump before the empty-text early return: a segment that produced
        # audio but no text ("audio came in, nothing recognized") is exactly a
        # case worth inspecting. dump_id only rides the wire when a final is
        # actually sent (text non-empty), keeping dump_id <-> bubble 1:1.
        dump_id: str | None = None
        if ctx.dumper is not None:
            dump_id = await ctx.dumper.write_final(
                seg_id=seg.id,
                pcm=segment,
                meta=self._segment_dump_meta(
                    text=text,
                    detected_lang=detected_lang,
                    audio_duration=audio_duration,
                    elapsed=elapsed,
                    rtf=rtf,
                    seg=seg,
                    primary_result=primary_result,
                    secondary_result=secondary_result,
                    hw_snapshot=hw_snapshot,
                    ctx=ctx,
                ),
            )

        if not text:
            return False

        payload: dict = {
            "type": "final",
            "text": text,
            "language": _public_language(detected_lang),
            "audio_b64": wav_b64,
            "duration_sec": audio_duration,
            "effective_hotwords": returned_effective_hotwords,
            "enrollment_applied": self._enrollment_applied_for_final(
                primary_result,
                ctx,
            ),
        }
        fallback_reason = self._enrollment_fallback_reason_for_final(
            primary_result,
            ctx,
        )
        if fallback_reason:
            payload["enrollment_fallback_reason"] = fallback_reason
        if dump_id:
            payload["dump_id"] = dump_id
        if seg.id:
            payload["id"] = seg.id
        if self._emit_timing:
            if seg.start_ms is not None:
                payload["bg_ms"] = seg.start_ms
            if seg.end_ms is not None:
                payload["ed_ms"] = seg.end_ms
        if seg.speaker_index is not None:
            payload["speaker_index"] = seg.speaker_index
        return await ctx.send_json(payload)

    # ------------------------------------------------------------------
    # Pseudo-streaming partial
    # ------------------------------------------------------------------

    async def handle_partial(self, snap: PartialSnapshot, ctx: SessionContext) -> None:
        cfg = ctx.cfg
        if not cfg.enable_pseudo_stream:
            return
        if not (cfg.enable_primary_asr or cfg.enable_secondary_asr):
            return

        voice_filter = filter_speech_for_asr(snap.pcm, cfg)
        if not voice_filter.evidence.accepted:
            logger.debug(
                "Partial ASR skipped by speech filter: id=%s reason=%s ratio=%.3f",
                snap.id or "n/a",
                voice_filter.evidence.reason,
                voice_filter.evidence.speech_ratio,
            )
            return
        snapshot = voice_filter.pcm
        audio_duration = len(snapshot) / SAMPLE_RATE
        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(snapshot)
        hw_snapshot = ctx.hotwords

        primary_res: object = None
        secondary_res: object = None

        # Pseudo-streaming is intentionally primary-only when the primary is
        # online: it should be pure vLLM raw-audio inference, without hotword
        # recall/bypass and without the secondary noise gate suppressing short
        # early snapshots. The secondary remains a fallback for deployments that
        # turn the primary off.
        if cfg.enable_primary_asr:
            primary_res = await asyncio.wait_for(
                query_audio_model(
                    wav_b64,
                    hotwords=[],
                    src_lang=ctx.src_lang,
                    audio_pcm=None,
                    audio_sample_rate=SAMPLE_RATE,
                    enrollment_wav_base64=ctx.enrollment_b64,
                    base_url=cfg.vllm_base_url,
                    model_name=cfg.vllm_model_name,
                    prompt_template=cfg.vllm_prompt_template,
                    timeout=cfg.asr_request_timeout,
                    runtime_config=cfg,
                    hotword_pool_id=ctx.hotword_pool_id,
                ),
                timeout=cfg.primary_asr_timeout,
            )
        elif cfg.enable_secondary_asr:
            secondary_res = await query_audio_model_secondary(
                wav_b64,
                hotwords=hw_snapshot,
                base_url=cfg.secondary_vllm_base_url,
                model_name=cfg.secondary_vllm_model_name,
                timeout=cfg.asr_request_timeout,
            )

        primary_result = None if isinstance(primary_res, Exception) else primary_res
        secondary_result = None if isinstance(secondary_res, Exception) else secondary_res

        if primary_result is None and secondary_result is None:
            return

        # Noise gate only applies when secondary is the actual partial source.
        if secondary_result is not None and primary_result is None:
            sec_text = str((secondary_result or {}).get("transcription") or "").strip()
            if not sec_text:
                logger.debug("Partial suppressed: secondary empty (noise gate)")
                return

        text, _ = self._select_text(primary_result, secondary_result, [], ctx)

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Partial ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration,
            elapsed,
            rtf,
            text[:80],
        )

        if not text:
            return

        # Record the partial history so the segment's dump can show how the
        # text evolved before the final settled (useful for "dropped word"
        # triage). Cheap, in-loop; no IO here.
        if ctx.dumper is not None:
            ctx.dumper.record_partial(snap.id, text)

        payload = {
            "type": "partial",
            "text": text,
            "language": _public_language(ctx.language),
            **({"id": snap.id} if snap.id else {}),
        }
        if ctx.enrollment_b64 or ctx.enrollment_id or ctx.enrollment_fallback_reason:
            payload["enrollment_applied"] = bool(ctx.enrollment_b64)
        if ctx.enrollment_fallback_reason:
            payload["enrollment_fallback_reason"] = ctx.enrollment_fallback_reason
        await ctx.send_json(payload)

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
        # Match the legacy behavior: only emit empty final after explicit stop
        # (not after raw socket close) when nothing was sent in this drain.
        if stopped and not sent_any_response:
            await ctx.send_json(
                {
                    "type": "final",
                    "text": "",
                    "language": _public_language(ctx.language),
                    "effective_hotwords": [],
                }
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    async def _refine_qwen_hotwords(
        text: str,
        *,
        language: str,
        hotwords: list[str],
    ) -> str:
        """Apply requested/recalled hotwords after Qwen3-ASR recognition.

        Qwen3-ASR deliberately has no prompt-hotword support. The public
        streaming contract still accepts hotwords, so the Qwen-only profile
        performs a conservative, fail-open correction before emitting final.
        """
        from ..clean_stream import SessionOptions, evaluate_cleanup_result, refine_text

        options = SessionOptions(
            language=language or "auto",
            cleanup_level="light",
            custom=list(hotwords),
        )
        try:
            refined = await refine_text(text, options, None)
            accepted, reason = evaluate_cleanup_result(
                text,
                refined,
                "light",
                options.glossary,
            )
            if accepted:
                return refined
            logger.warning("Qwen streaming hotword refinement rejected: %s", reason)
        except Exception as exc:  # noqa: BLE001 - raw ASR must remain usable
            logger.warning("Qwen streaming hotword refinement failed: %s", exc)
        return text

    @staticmethod
    def _result_text(result: object) -> str:
        if isinstance(result, dict):
            return str(result.get("transcription") or "")
        return ""

    @staticmethod
    def _result_raw(result: object) -> str:
        if isinstance(result, dict):
            return str(result.get("raw_text") or "")
        return ""

    def _segment_dump_meta(
        self,
        *,
        text: str,
        detected_lang: str | None,
        audio_duration: float,
        elapsed: float,
        rtf: float,
        seg: SegmentReady,
        primary_result: object,
        secondary_result: object,
        hw_snapshot: list[str],
        ctx: SessionContext,
    ) -> dict:
        """Assemble the per-segment dump record (JSON-serializable)."""
        cfg = ctx.cfg
        reported = (
            list(primary_result.get("reported_hotwords") or [])
            if isinstance(primary_result, dict)
            else []
        )
        return {
            "text": text,
            "detected_language": detected_lang,
            "language": ctx.language,
            "src_lang": ctx.src_lang,
            "audio": {
                "duration_sec": round(audio_duration, 3),
                "sample_rate": SAMPLE_RATE,
                "num_samples": int(len(seg.pcm)),
            },
            "timing": {
                "infer_elapsed_sec": round(elapsed, 3),
                "rtf": round(rtf, 3),
                "bg_ms": seg.start_ms,
                "ed_ms": seg.end_ms,
                "is_stop_flush": seg.is_stop_flush,
                "speaker_index": seg.speaker_index,
            },
            "asr": {
                "primary_text": self._result_text(primary_result),
                "primary_raw": self._result_raw(primary_result),
                "secondary_text": self._result_text(secondary_result),
                "secondary_raw": self._result_raw(secondary_result),
                "reported_hotwords": reported,
                "hotwords_snapshot": list(hw_snapshot or []),
                "hotword_pool_id": ctx.hotword_pool_id,
            },
            "model": {
                "vllm_base_url": cfg.vllm_base_url,
                "vllm_model_name": cfg.vllm_model_name,
                "vllm_prompt_template": cfg.vllm_prompt_template,
                "secondary_vllm_base_url": cfg.secondary_vllm_base_url,
                "secondary_vllm_model_name": cfg.secondary_vllm_model_name,
            },
            "flags": {
                "enable_primary_asr": cfg.enable_primary_asr,
                "enable_secondary_asr": cfg.enable_secondary_asr,
                "enable_dual_asr_fusion": cfg.enable_dual_asr_fusion,
                "enable_hotword_recall": cfg.enable_hotword_recall,
                "enable_encoder_bypass": cfg.enable_encoder_bypass,
                "k2_enabled": cfg.k2_enabled,
                "enable_pseudo_stream": cfg.enable_pseudo_stream,
            },
        }

    async def _dual_asr(
        self,
        wav_b64: str,
        hw_snapshot: list[str],
        ctx: SessionContext,
        *,
        audio_pcm=None,
    ) -> tuple:
        cfg = ctx.cfg
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(
                wav_b64,
                hotwords=hw_snapshot,
                base_url=cfg.secondary_vllm_base_url,
                model_name=cfg.secondary_vllm_model_name,
                timeout=cfg.asr_request_timeout,
            )
        )
        primary_task = None
        if cfg.enable_primary_asr:
            primary_task = asyncio.create_task(
                asyncio.wait_for(
                    query_audio_model(
                        wav_b64,
                        hotwords=hw_snapshot if audio_pcm is not None else [],
                        src_lang=ctx.src_lang,
                        audio_pcm=audio_pcm,
                        audio_sample_rate=SAMPLE_RATE,
                        enrollment_wav_base64=ctx.enrollment_b64,
                        base_url=cfg.vllm_base_url,
                        model_name=cfg.vllm_model_name,
                        prompt_template=cfg.vllm_prompt_template,
                        timeout=cfg.asr_request_timeout,
                        runtime_config=cfg,
                        hotword_pool_id=ctx.hotword_pool_id,
                        enrollment_id=ctx.enrollment_id,
                        enrollment_scope_id=cfg.hotword_pool_id,
                        session_id=ctx.session_id,
                        gateway_trace_id=ctx.gateway_trace_id,
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

        secondary_text = str((secondary_res or {}).get("transcription") or "").strip()
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
            detected_lang = primary_result.get("detected_language") or ctx.language
        elif secondary_result and not primary_result:
            text = str(secondary_result.get("transcription") or "").strip()
        else:
            fusion_hotwords = self._fusion_hotwords(primary_result, hw_snapshot)
            fused = choose_fused_result(
                primary_result,
                secondary_result,
                hotwords=fusion_hotwords,
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

    @staticmethod
    def _fusion_hotwords(primary_result, fallback: list[str]) -> list[str]:
        if primary_result:
            reported = primary_result.get("reported_hotwords") or []
            if reported:
                return [str(word) for word in reported]
        return fallback

    @staticmethod
    def _effective_hotwords_for_final(primary_result, fallback: list[str]) -> list[str]:
        if isinstance(primary_result, dict):
            reported = primary_result.get("reported_hotwords") or []
            if reported:
                return [str(word) for word in reported]
        return list(fallback or [])

    @staticmethod
    def _rag_recalled_hotwords_for_final(primary_result) -> list[str]:
        if isinstance(primary_result, dict):
            return [str(word) for word in primary_result.get("effective_hotwords") or []]
        return []

    @staticmethod
    def _enrollment_applied_for_final(primary_result, ctx: SessionContext) -> bool:
        if isinstance(primary_result, dict) and "enrollment_applied" in primary_result:
            return bool(primary_result.get("enrollment_applied"))
        return bool(ctx.enrollment_b64)

    @staticmethod
    def _enrollment_fallback_reason_for_final(
        primary_result,
        ctx: SessionContext,
    ) -> str:
        if isinstance(primary_result, dict):
            reason = str(primary_result.get("enrollment_fallback_reason") or "").strip()
            if reason:
                return reason
        if ctx.enrollment_fallback_reason:
            return ctx.enrollment_fallback_reason
        if ctx.enrollment_id and not ctx.enrollment_b64:
            return "not_found"
        return ""
