import re
from typing import Any, TypedDict

from ..config import (
    ASR_REQUEST_TIMEOUT,
    SECONDARY_VLLM_BASE_URL,
    SECONDARY_VLLM_MODEL_NAME,
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
)
from ..http_client import get_client


class ASRResult(TypedDict):
    transcription: str
    reported_hotwords: list[str]
    raw_text: str
    detected_language: str | None


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


def build_primary_prompt(hotwords: list[str], src_lang: str) -> str:
    hw_str = ",".join(hotwords) if hotwords else ""
    return (
        "Transcribe the following audio.\n"
        f"Language: {src_lang}\n"
        f"Hotwords: {hw_str}"
    )


def build_single_turn_messages(prompt_text: str, audio_wav_base64: str) -> list[dict]:
    content: list[dict[str, Any]] = []
    if str(prompt_text or "").strip():
        content.append({"type": "text", "text": prompt_text})
    content.append(
        {
            "type": "input_audio",
            "input_audio": {
                "data": audio_wav_base64,
                "format": "wav",
            },
        }
    )
    return [
        {
            "role": "user",
            "content": content,
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


def parse_model_output(raw_text: str) -> ASRResult:
    """Parse model output with ``Language:`` / ``Hotwords:`` / ``Transcription:`` lines."""
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

    return ASRResult(
        transcription=transcription,
        reported_hotwords=reported_hotwords,
        raw_text=raw,
        detected_language=detected_language,
    )


async def _query_audio_model_by_endpoint(
    audio_wav_base64: str,
    *,
    base_url: str,
    model_name: str,
    hotwords: list[str] | None,
    src_lang: str,
    audio_only: bool,
    timeout: float = ASR_REQUEST_TIMEOUT,
) -> ASRResult:
    client = get_client()
    prompt_text = (
        "" if audio_only else build_primary_prompt(hotwords or [], src_lang)
    )
    base = base_url.rstrip("/")

    resp = await client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": build_single_turn_messages(prompt_text, audio_wav_base64),
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
    src_lang: str = "N/A",
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
) -> ASRResult:
    return await _query_audio_model_by_endpoint(
        audio_wav_base64,
        base_url=base_url or VLLM_BASE_URL,
        model_name=model_name or VLLM_MODEL_NAME,
        hotwords=hotwords,
        src_lang=src_lang,
        audio_only=False,
        timeout=timeout if timeout is not None else ASR_REQUEST_TIMEOUT,
    )


async def query_audio_model_secondary(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
) -> ASRResult:
    _ = hotwords
    return await _query_audio_model_by_endpoint(
        audio_wav_base64,
        base_url=base_url or SECONDARY_VLLM_BASE_URL,
        model_name=model_name or SECONDARY_VLLM_MODEL_NAME,
        hotwords=[],
        src_lang="N/A",
        audio_only=True,
        timeout=timeout if timeout is not None else ASR_REQUEST_TIMEOUT,
    )
