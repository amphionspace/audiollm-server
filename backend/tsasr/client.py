"""vLLM client for the TS-ASR task.

Uses the OpenAI-compatible chat completions API, assembling the two-audio
``content`` list via :func:`backend.tsasr.prompt.build_tsasr_content`. The
parser reuses :func:`backend.asr.client.parse_model_output` because the
Amphion model is only trained to emit ``{transcription}`` for the ts_asr
task; ``Language:`` / ``Hotwords:`` prefixes (if they ever appear) will be
handled transparently by the existing parser's fall-back branches.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..asr.client import ASRResult, parse_model_output
from ..config import ASR_REQUEST_TIMEOUT, VLLM_BASE_URL, VLLM_MODEL_NAME
from ..http_client import get_client
from .prompt import build_tsasr_content


class TsAsrResult(TypedDict):
    """Wire-level TS-ASR result mirroring :class:`ASRResult` plus meta."""

    transcription: str
    raw_text: str
    detected_language: str | None
    enrollment_duration_sec: float | None


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


async def query_tsasr_model(
    mixed_wav_base64: str,
    enrollment_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    voice_traits: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
    enrollment_duration_sec: float | None = None,
) -> TsAsrResult:
    """Run one TS-ASR inference against the vLLM OpenAI-compatible endpoint.

    ``base_url`` / ``model_name`` are resolved by the caller (the task engine
    handles the ``tsasr_* -> vllm_*`` fallback). An explicit ``None`` here
    falls back to the module-level primary ASR endpoint so that this client
    remains usable for ad-hoc scripts.
    """
    client = get_client()
    content = build_tsasr_content(
        enrollment_wav_base64,
        mixed_wav_base64,
        hotwords=hotwords,
        voice_traits=voice_traits,
    )
    messages = [{"role": "user", "content": content}]
    payload: dict[str, Any] = {
        "model": model_name or VLLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": int(max_tokens) if max_tokens else 512,
        "temperature": 0,
    }

    base = (base_url or VLLM_BASE_URL).rstrip("/")
    resp = await client.post(
        f"{base}/v1/chat/completions",
        json=payload,
        timeout=timeout if timeout is not None else ASR_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    parsed: ASRResult = parse_model_output(raw_text)

    return TsAsrResult(
        transcription=parsed["transcription"],
        raw_text=parsed["raw_text"],
        detected_language=parsed["detected_language"],
        enrollment_duration_sec=enrollment_duration_sec,
    )
