import hashlib
import json
import logging
import re
import time
from typing import Any, TypedDict

import numpy as np
from pypinyin import Style, lazy_pinyin

from ..config import SAMPLE_RATE, Config, default_config
from ..http_client import get_client
from .prompt_templates import audio_item
from .prompt_templates import build_primary_messages as _build_primary_messages
from .recall import recall_audio

logger = logging.getLogger(__name__)

_CHINESE_WORD_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")


def _hotwords_hash(words: list[str]) -> str:
    encoded = json.dumps(words, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _hotwords_preview(words: list[str], *, limit: int = 50) -> dict[str, object]:
    payload: dict[str, object] = {
        "count": len(words),
        "hotwords": list(words[:limit]),
    }
    if len(words) > limit:
        payload["hotwords_truncated"] = len(words) - limit
        payload["hotwords_hash"] = _hotwords_hash(words)
    return payload


def hotword_hits_in_text(hotwords: list[str], text: str) -> list[str]:
    final_text = str(text or "")
    return [word for word in hotwords if word and word in final_text]


def diagnostic_hotword_misses(hotwords: list[str], text: str) -> list[str]:
    hits = set(hotword_hits_in_text(hotwords, text))
    misses = [word for word in hotwords if word and word not in hits]
    return [
        word
        for word in misses
        if len(word) <= 3 or _hotword_pronunciation_key(word) is not None
    ][:50]


class ASRResult(TypedDict):
    transcription: str
    reported_hotwords: list[str]
    raw_text: str
    detected_language: str | None


def build_primary_messages(
    target_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    enrollment_wav_base64: str | None = None,
    enrollment_audio_embeds_b64: str | None = None,
    enrollment_audio_embeds_uuid: str | None = None,
    audio_embeds_b64: str | None = None,
    audio_embeds_uuid: str | None = None,
    template: str | None = None,
) -> list[dict]:
    """Build primary ASR messages for the selected model prompt template."""
    return _build_primary_messages(
        target_wav_base64,
        hotwords=hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
        enrollment_audio_embeds_b64=enrollment_audio_embeds_b64,
        enrollment_audio_embeds_uuid=enrollment_audio_embeds_uuid,
        audio_embeds_b64=audio_embeds_b64,
        audio_embeds_uuid=audio_embeds_uuid,
        template=template or default_config.vllm_prompt_template,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content or "")


def build_audio_only_messages(audio_wav_base64: str) -> list[dict]:
    """Single-audio prompt without any text — used by the Qwen3 secondary
    path, which is trained as a pure ASR model and ignores text guidance."""
    return [
        {
            "role": "user",
            "content": [audio_item(audio_wav_base64)],
        }
    ]


def _parse_hotwords_field(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "-"}:
        return []
    return [item.strip() for item in re.split(r"[,，;；]", text) if item.strip()]


def _parse_language_field(value: str) -> str | None:
    v = str(value or "").strip()
    if not v:
        return None
    if v.lower() in {"n/a", "na", "none", "null", "-"}:
        return None
    return v


def _hotword_pronunciation_key(word: str) -> tuple[str, ...] | None:
    """Return a whole-word pinyin key for pure Chinese hotwords."""
    if not _CHINESE_WORD_RE.fullmatch(word):
        return None
    syllables = lazy_pinyin(
        word,
        style=Style.NORMAL,
        errors="ignore",
        strict=False,
    )
    if len(syllables) != len(word) or not all(syllables):
        return None
    return tuple(syllables)


def merge_recalled_and_custom_hotwords(
    recalled: list[str] | None,
    custom: list[str] | None,
    *,
    custom_limit: int,
) -> list[str]:
    """Prefer request-local hotwords and suppress same-pronunciation recalls."""
    merged: list[str] = []
    seen: set[str] = set()
    custom_pronunciations: set[tuple[str, ...]] = set()

    def add_word(raw: object) -> bool:
        word = str(raw or "").strip()
        if not word or word in seen:
            return False
        seen.add(word)
        merged.append(word)
        return True

    remaining = max(int(custom_limit), 0)
    if remaining > 0:
        for word in custom or []:
            before = len(merged)
            add_word(word)
            if len(merged) > before:
                key = _hotword_pronunciation_key(merged[-1])
                if key is not None:
                    custom_pronunciations.add(key)
                remaining -= 1
                if remaining <= 0:
                    break
    for word in recalled or []:
        normalized = str(word or "").strip()
        key = _hotword_pronunciation_key(normalized)
        if key is not None and key in custom_pronunciations:
            continue
        add_word(normalized)
    return merged


def _postprocess_asr_text(text: str) -> str:
    """Normalize provider-specific wrappers to plain transcription text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*<asr_text>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*[:：-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """Collapse pathological decode loops while leaving normal text untouched."""
    source = str(text or "")
    if len(source) <= threshold:
        return source

    fixed = source
    max_unit = min(16, max(1, len(fixed) // (threshold + 1)))
    for unit_len in range(1, max_unit + 1):
        out: list[str] = []
        i = 0
        changed = False
        while i < len(fixed):
            unit = fixed[i : i + unit_len]
            if len(unit) < unit_len:
                out.append(fixed[i:])
                break
            count = 1
            j = i + unit_len
            while fixed[j : j + unit_len] == unit:
                count += 1
                j += unit_len
            if count > threshold:
                out.append(unit)
                i = j
                changed = True
            else:
                out.append(fixed[i])
                i += 1
        if changed:
            fixed = "".join(out)
    return fixed


def parse_model_output(
    raw_text: str,
    *,
    enable_repetition_fix: bool | None = None,
) -> ASRResult:
    """Parse model output wrappers and normalize to plain transcription text."""
    raw = str(raw_text or "").strip()
    if not raw:
        return ASRResult(
            transcription="",
            reported_hotwords=[],
            raw_text="",
            detected_language=None,
        )

    normalized = raw.replace("\\r\\n", "\n").replace("\\n", "\n")

    lang_m = re.search(
        r"(?:^|\n)\s*language\s*:\s*([^\n]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    detected_language = (
        _parse_language_field(lang_m.group(1)) if lang_m else None
    )
    if detected_language is None:
        qwen_lang_m = re.search(
            r"(?:^|\n)\s*language\s+([A-Za-z\u4e00-\u9fff_-]+)\s*<asr_text>",
            normalized,
            flags=re.IGNORECASE,
        )
        detected_language = (
            _parse_language_field(qwen_lang_m.group(1)) if qwen_lang_m else None
        )

    hw_m = re.search(
        r"(?:^|\n)\s*hotwords\s*:\s*(.+?)(?=\n\s*(?:language|transcription)\s*:|\Z)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not hw_m:
        hw_m = re.search(
            r"(?:^|\n)\s*hotwords\s*:\s*(.+?)(?=\n\s*[A-Za-z_]+\s*:|\Z)",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    reported_hotwords = (
        _parse_hotwords_field(hw_m.group(1)) if hw_m else []
    )

    hm = re.search(r"(?i)hotwords\s*:", normalized)
    tm = re.search(r"(?i)transcription\s*:", normalized)
    h_start = hm.start() if hm else -1
    t_start = tm.start() if tm else -1

    transcription = ""
    if tm:
        if h_start >= 0 and h_start < t_start:
            m_tr = re.search(
                r"(?:^|\n)\s*transcription\s*:\s*(.*)\Z",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            transcription = m_tr.group(1).strip() if m_tr else ""
        else:
            m_tr = re.search(
                r"(?:^|\n)\s*transcription\s*:\s*(.+?)(?=\n\s*hotwords\s*:|\Z)",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            transcription = (
                m_tr.group(1).strip() if m_tr else normalized.strip()
            )
    else:
        transcription = normalized.strip()

    transcription = _postprocess_asr_text(transcription)
    if enable_repetition_fix is None:
        enable_repetition_fix = default_config.enable_asr_repetition_fix
    if enable_repetition_fix:
        transcription = detect_and_fix_repetitions(transcription)

    return ASRResult(
        transcription=transcription,
        reported_hotwords=reported_hotwords,
        raw_text=raw,
        detected_language=detected_language,
    )


async def _post_chat(
    messages: list[dict],
    *,
    base_url: str,
    model_name: str,
    timeout: float,
) -> ASRResult:
    client = get_client()
    base = base_url.rstrip("/")
    resp = await client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": 512,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    return parse_model_output(raw_text)


async def query_audio_model(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    src_lang: str = "N/A",  # accepted for callsite compatibility, ignored
    audio_pcm: np.ndarray | None = None,
    audio_sample_rate: int = SAMPLE_RATE,
    enrollment_wav_base64: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    prompt_template: str | None = None,
    timeout: float | None = None,
    runtime_config: Config | None = None,
    hotword_pool_id: str | None = None,
    enrollment_id: str | None = None,
    enrollment_scope_id: str | None = None,
    session_id: str = "",
    gateway_trace_id: str = "",
) -> ASRResult:
    """Primary ASR call.

    ``src_lang`` is intentionally not forwarded into the prompt. The
    primary model's prompt format is selected by ``prompt_template`` (or
    the configured default) so Amphion 4B and 1.7B can coexist without
    duplicating call sites.
    """
    _ = src_lang  # noqa: F841 — preserved for compatibility, see docstring
    cfg = runtime_config or default_config
    template = prompt_template or cfg.vllm_prompt_template
    resolved_hotword_pool_id = hotword_pool_id
    request_hotwords = list(hotwords or [])
    effective_hotwords = request_hotwords
    recalled_hotwords: list[str] = []
    audio_embeds_b64: str | None = None
    audio_embeds_uuid: str | None = None
    enrollment_audio_embeds_b64: str | None = None
    enrollment_audio_embeds_uuid: str | None = None
    audio_embeds_bypass_used = False
    enrollment_embeds_bypass_used = False
    enrollment_fallback_reason = ""
    want_enrollment_bypass = False
    is_final_path = audio_pcm is not None

    if cfg.enable_hotword_recall and audio_pcm is not None:
        # The recall service owns the large per-user pool. Request-local
        # hotwords are still useful for one-off biasing, but must stay bounded
        # so clients cannot accidentally paste a huge pool into every prompt.
        effective_hotwords = merge_recalled_and_custom_hotwords(
            [],
            request_hotwords,
            custom_limit=cfg.recall_custom_hotword_limit,
        )
        want_enrollment_bypass = (
            cfg.enable_triton_enrollment_store
            and cfg.enable_enrollment_embedding_bypass
            and template == "amphion_asr_1.7b"
            and bool(enrollment_id)
            and enrollment_wav_base64 is None
        )
        if (
            cfg.enable_triton_enrollment_store
            and enrollment_id
            and enrollment_wav_base64 is None
            and not want_enrollment_bypass
        ):
            enrollment_fallback_reason = "incompatible"
        want_bypass = (
            cfg.enable_encoder_bypass
            and template == "amphion_asr_1.7b"
            and (enrollment_wav_base64 is None)
        )
        recall_t0 = time.monotonic()
        try:
            recalled = await recall_audio(
                audio_pcm,
                cfg,
                sample_rate=audio_sample_rate,
                want_audio_embeds=want_bypass,
                hotword_pool_id=resolved_hotword_pool_id,
                enrollment_id=enrollment_id if want_enrollment_bypass else None,
                enrollment_scope_id=enrollment_scope_id if want_enrollment_bypass else None,
                want_enrollment_audio_embeds=want_enrollment_bypass,
            )
            recall_latency_ms = (time.monotonic() - recall_t0) * 1000.0
            recalled_hotwords = list(recalled.words)
            effective_hotwords = merge_recalled_and_custom_hotwords(
                recalled_hotwords,
                request_hotwords,
                custom_limit=cfg.recall_custom_hotword_limit,
            )
            if want_bypass and recalled.audio_embeds_b64:
                audio_embeds_b64 = recalled.audio_embeds_b64
                audio_embeds_uuid = recalled.uuid
                audio_embeds_bypass_used = True
            recalled_enrollment_embeds = getattr(
                recalled,
                "enrollment_audio_embeds_b64",
                None,
            )
            if want_enrollment_bypass and recalled_enrollment_embeds:
                enrollment_audio_embeds_b64 = recalled_enrollment_embeds
                enrollment_owner = (
                    enrollment_scope_id
                    or resolved_hotword_pool_id
                    or cfg.hotword_pool_id
                )
                enrollment_audio_embeds_uuid = (
                    f"triton-enrollment-{enrollment_owner}-{enrollment_id}"
                )
                enrollment_embeds_bypass_used = True
                enrollment_fallback_reason = ""
            elif want_enrollment_bypass:
                enrollment_fallback_reason = "not_found"
            logger.info(
                "ASR recall result: session_id=%s gateway_trace_id=%s "
                "hotword_pool_id=%s recall_top_k=%d recalled_count=%d "
                "recalled=%s recall_latency_ms=%.1f "
                "audio_embeds_requested=%s audio_embeds_returned=%s "
                "enrollment_embeds_requested=%s enrollment_embeds_returned=%s",
                session_id or "n/a",
                gateway_trace_id or "n/a",
                resolved_hotword_pool_id or cfg.hotword_pool_id,
                cfg.recall_top_k,
                len(recalled_hotwords),
                _hotwords_preview(recalled_hotwords, limit=50),
                recall_latency_ms,
                want_bypass,
                bool(recalled.audio_embeds_b64),
                want_enrollment_bypass,
                bool(recalled_enrollment_embeds),
            )
        except Exception as exc:
            recall_latency_ms = (time.monotonic() - recall_t0) * 1000.0
            if want_enrollment_bypass:
                enrollment_fallback_reason = "upstream_unavailable"
            logger.warning(
                "ASR recall failed: session_id=%s gateway_trace_id=%s "
                "hotword_pool_id=%s recall_top_k=%d reason=%r "
                "recall_latency_ms=%.1f",
                session_id or "n/a",
                gateway_trace_id or "n/a",
                resolved_hotword_pool_id or cfg.hotword_pool_id,
                cfg.recall_top_k,
                exc,
                recall_latency_ms,
            )

    if is_final_path:
        logger.info(
            "ASR prompt hotwords: session_id=%s gateway_trace_id=%s "
            "custom_hotwords_count=%d recalled_hotwords_count=%d "
            "effective_hotwords_count=%d effective=%s prompt_has_hotwords=%s "
            "audio_embeds_bypass_requested=%s audio_embeds_bypass_used=%s",
            session_id or "n/a",
            gateway_trace_id or "n/a",
            len(request_hotwords),
            len(recalled_hotwords),
            len(effective_hotwords),
            _hotwords_preview(effective_hotwords, limit=50),
            bool(effective_hotwords),
            (
                cfg.enable_encoder_bypass
                and template == "amphion_asr_1.7b"
                and enrollment_wav_base64 is None
            ),
            audio_embeds_bypass_used,
        )

    messages = build_primary_messages(
        audio_wav_base64,
        hotwords=effective_hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
        enrollment_audio_embeds_b64=enrollment_audio_embeds_b64,
        enrollment_audio_embeds_uuid=enrollment_audio_embeds_uuid,
        audio_embeds_b64=audio_embeds_b64,
        audio_embeds_uuid=audio_embeds_uuid,
        template=template,
    )
    request_base_url = base_url or cfg.vllm_base_url
    request_model_name = model_name or cfg.vllm_model_name
    request_timeout = timeout if timeout is not None else cfg.asr_request_timeout
    try:
        result = await _post_chat(
            messages,
            base_url=request_base_url,
            model_name=request_model_name,
            timeout=request_timeout,
        )
    except Exception:
        if not audio_embeds_b64:
            raise
        logger.warning("Primary ASR audio_embeds request failed; retrying raw audio")
        audio_embeds_bypass_used = False
        if enrollment_embeds_bypass_used:
            enrollment_fallback_reason = "upstream_unavailable"
        enrollment_embeds_bypass_used = False
        fallback_messages = build_primary_messages(
            audio_wav_base64,
            hotwords=effective_hotwords,
            enrollment_wav_base64=enrollment_wav_base64,
            template=template,
        )
        result = await _post_chat(
            fallback_messages,
            base_url=request_base_url,
            model_name=request_model_name,
            timeout=request_timeout,
        )
    if effective_hotwords:
        result["reported_hotwords"] = effective_hotwords
    if is_final_path:
        result["effective_hotwords"] = list(recalled_hotwords)
        enrollment_applied = bool(enrollment_wav_base64 or enrollment_embeds_bypass_used)
        if enrollment_wav_base64 or enrollment_id:
            result["enrollment_applied"] = enrollment_applied
            if not enrollment_applied:
                result["enrollment_fallback_reason"] = (
                    enrollment_fallback_reason or "not_found"
                )
        text = str(result.get("transcription") or "")
        hits = hotword_hits_in_text(effective_hotwords, text)
        logger.info(
            "ASR model final diagnostic: session_id=%s gateway_trace_id=%s "
            "final_text=%r effective_hotwords_count=%d "
            "effective_hotword_hits=%s audio_embeds_bypass_used=%s",
            session_id or "n/a",
            gateway_trace_id or "n/a",
            text,
            len(effective_hotwords),
            hits,
            audio_embeds_bypass_used or enrollment_embeds_bypass_used,
        )
        misses = diagnostic_hotword_misses(effective_hotwords, text)
        if misses:
            logger.warning(
                "ASR hotword miss: session_id=%s gateway_trace_id=%s hotword_miss=%s",
                session_id or "n/a",
                gateway_trace_id or "n/a",
                misses,
            )
    return result


async def query_audio_model_secondary(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
) -> ASRResult:
    _ = hotwords
    messages = build_audio_only_messages(audio_wav_base64)
    return await _post_chat(
        messages,
        base_url=base_url or default_config.secondary_vllm_base_url,
        model_name=model_name or default_config.secondary_vllm_model_name,
        timeout=timeout if timeout is not None else default_config.asr_request_timeout,
    )
