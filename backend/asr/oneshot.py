"""Single-clip dual-ASR inference shared by REST upload and offline jobs.

Extracted from ``backend/main.py``'s former ``_run_dual_asr_upload`` so the
"primary + optional secondary in parallel, fuse, ITN" semantics exist exactly
once for every non-streaming caller (``/api/asr/upload``,
``/api/audio/analyze``, and the long-audio transcription pipeline). The
streaming engine keeps its own copy of the orchestration because it also
manages partials/noise gates; the fusion / ITN building blocks themselves are
shared via :mod:`backend.asr.fusion` and :mod:`backend.asr.itn`.

Framework-free by design: failures surface as :class:`OneshotAsrError`, and
the route layer decides how to map that onto HTTP.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from ..audio.utils import pcm_to_wav_base64
from ..audio.vad import trim_long_silence_for_asr
from ..config import SAMPLE_RATE, Config
from .client import query_audio_model, query_audio_model_secondary
from .fusion import choose_fused_result
from .itn import normalize_final_text

logger = logging.getLogger(__name__)

__all__ = ["OneshotAsrError", "model_result_payload", "run_oneshot_asr"]


class OneshotAsrError(Exception):
    """Every configured ASR model failed for this clip."""

    def __init__(self, *, primary: object | None, secondary: object | None) -> None:
        super().__init__("all configured ASR models failed")
        self.primary = primary
        self.secondary = secondary

    def to_detail(self) -> dict:
        return {
            "message": "all configured ASR models failed",
            "primary": model_result_payload(self.primary),
            "secondary": model_result_payload(self.secondary),
        }


def model_result_payload(result: object | None) -> dict | None:
    if result is None:
        return None
    if isinstance(result, Exception):
        return {
            "error": str(result),
            "error_type": result.__class__.__name__,
        }
    if isinstance(result, dict):
        return dict(result)
    return {"raw": str(result)}


async def run_oneshot_asr(
    wav_b64: str,
    *,
    cfg: Config,
    hotwords: list[str],
    language: str,
    audio_pcm: np.ndarray | None = None,
    enrollment_b64: str | None = None,
    enrollment_id: str | None = None,
    enrollment_scope_id: str | None = None,
    hotword_pool_id: str | None = None,
) -> dict:
    """Transcribe one clip with the configured primary/secondary models.

    Returns ``{text, language, raw_text, primary, secondary, fusion}`` where
    ``text`` already went through final-only display transforms (ITN + plate
    normalization). Raises :class:`OneshotAsrError` when every configured
    model failed.
    """
    if audio_pcm is not None:
        original_duration = len(audio_pcm) / SAMPLE_RATE
        trim = trim_long_silence_for_asr(audio_pcm, cfg)
        if trim.removed_ranges:
            logger.info(
                "One-shot ASR silence removal: original=%.2fs removed=%.2fs "
                "ranges=%d trimmed=%.2fs threshold=%.2fs",
                original_duration,
                trim.removed_sec,
                trim.removed_ranges,
                len(trim.pcm) / SAMPLE_RATE,
                cfg.asr_silence_removal_threshold_sec,
            )
            audio_pcm = trim.pcm
            wav_b64 = pcm_to_wav_base64(audio_pcm)

    primary_task = None
    secondary_task = None
    if cfg.enable_primary_asr:
        primary_task = asyncio.create_task(
            asyncio.wait_for(
                query_audio_model(
                    wav_b64,
                    hotwords=hotwords,
                    src_lang=language or "N/A",
                    audio_pcm=audio_pcm,
                    enrollment_wav_base64=enrollment_b64,
                    base_url=cfg.vllm_base_url,
                    model_name=cfg.vllm_model_name,
                    prompt_template=cfg.vllm_prompt_template,
                    timeout=cfg.asr_request_timeout,
                    runtime_config=cfg,
                    hotword_pool_id=hotword_pool_id,
                    enrollment_id=enrollment_id,
                    enrollment_scope_id=enrollment_scope_id,
                ),
                timeout=cfg.primary_asr_timeout,
            )
        )
    # Secondary (Qwen3) keeps single-audio prompting regardless of enrollment;
    # it is trained as a plain ASR model with no target-speaker channel, so
    # forcing enrollment audio in front of the mixed clip would push the model
    # out of distribution. The fusion stage still benefits from Qwen's
    # parallel transcription as a sanity check on the primary's output.
    #
    # One-shot calls have no partial channel, so the secondary only earns its
    # keep when fusion is on. Gating on `enable_dual_asr_fusion` here (rather
    # than `enable_secondary_asr`) lets operators keep the secondary online
    # for streaming partials while skipping it on one-shot inference.
    if cfg.enable_dual_asr_fusion:
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(
                wav_b64,
                hotwords=hotwords,
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
        except Exception as err:  # noqa: BLE001 - preserve failure details
            primary_res = err
            logger.warning("Primary ASR failed: %s", err)
    if secondary_task is not None:
        try:
            secondary_res = await secondary_task
        except Exception as err:  # noqa: BLE001
            secondary_res = err
            logger.warning("Secondary ASR failed: %s", err)

    primary_result = None if isinstance(primary_res, Exception) else primary_res
    secondary_result = None if isinstance(secondary_res, Exception) else secondary_res
    if primary_result is None and secondary_result is None:
        raise OneshotAsrError(primary=primary_res, secondary=secondary_res)

    detected_lang = language or ""
    fusion_payload: dict | None = None
    if primary_result and not secondary_result:
        text = str(primary_result.get("transcription") or "").strip()
        detected_lang = primary_result.get("detected_language") or detected_lang
    elif secondary_result and not primary_result:
        text = str(secondary_result.get("transcription") or "").strip()
    else:
        fusion_hotwords = _fusion_hotwords(primary_result, hotwords)
        fusion_payload = choose_fused_result(
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
        text = str(fusion_payload.get("text") or "").strip()
        if primary_result and primary_result.get("detected_language"):
            detected_lang = primary_result["detected_language"]

    # Final-only display transform (ITN + plate normalization), matching the
    # streaming engines so REST and WS clients see the same written form.
    if text:
        text = normalize_final_text(text, detected_lang, cfg)

    raw_text = ""
    if primary_result:
        raw_text = str(primary_result.get("raw_text") or "")
    elif secondary_result:
        raw_text = str(secondary_result.get("raw_text") or "")

    return {
        "text": text,
        "language": detected_lang,
        "raw_text": raw_text,
        "effective_hotwords": _rag_recalled_hotwords(primary_result),
        "primary": model_result_payload(primary_res),
        "secondary": model_result_payload(secondary_res),
        "fusion": fusion_payload,
    }


def _fusion_hotwords(primary_result, fallback: list[str]) -> list[str]:
    if primary_result:
        reported = primary_result.get("reported_hotwords") or []
        if reported:
            return [str(word) for word in reported]
    return fallback


def _rag_recalled_hotwords(primary_result) -> list[str]:
    if isinstance(primary_result, dict):
        return [str(word) for word in primary_result.get("effective_hotwords") or []]
    return []
