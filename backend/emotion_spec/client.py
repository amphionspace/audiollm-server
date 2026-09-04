"""vLLM (OpenAI-compatible) client for the AmphionSPEC paralinguistic model.

Mirrors :mod:`backend.emotion.client` but points at an independent base
URL/model name and uses the emotion prompt set (``ser`` / ``sec``). All
post-hoc parsing helpers (``_content_to_text``, ``_strip_code_fence``,
``_match_taxonomy``, SER taxonomy itself) are model-agnostic — we import
them from :mod:`backend.emotion.client` instead of duplicating, so any
taxonomy/parsing fix lands in both clients at once.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict, cast

from ..config import default_config
from ..emotion.client import (  # noqa: F401 — re-export taxonomy for callers
    SER_TAXONOMY,
    _content_to_text,
    _match_taxonomy,
    _strip_code_fence,
)
from ..http_client import get_client
from .prompt import DEFAULT_MODE, EmotionSpecMode, get_prompt, normalize_mode

logger = logging.getLogger(__name__)


class EmotionSpecResult(TypedDict):
    """Normalized SPEC-model output.

    Same shape as :class:`backend.emotion.client.EmotionResult` so the
    frontend can render either result through one code path. The public
    ``mode`` is always ``ser`` or ``sec``.
    """

    mode: EmotionSpecMode
    label: str
    text: str
    raw_text: str


def _build_messages(audio_wav_base64: str, mode: EmotionSpecMode) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": get_prompt(mode)},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_wav_base64,
                        "format": "wav",
                    },
                },
            ],
        }
    ]


def _parse_ser(raw: str) -> EmotionSpecResult:
    """Parse SER output: model is trained to emit exactly one taxonomy label."""
    candidate = _strip_code_fence(raw).strip()

    label = _match_taxonomy(candidate)
    if not label:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                for key in ("label", "emotion", "class"):
                    val = parsed.get(key)
                    if isinstance(val, str):
                        label = _match_taxonomy(val)
                        if label:
                            break
            elif isinstance(parsed, str):
                label = _match_taxonomy(parsed)
        except json.JSONDecodeError:
            pass

    if not label:
        logger.warning("Could not map SPEC/SER output to taxonomy: %.200s", raw)

    return EmotionSpecResult(mode="ser", label=label, text=label, raw_text=raw)


def _parse_sec(raw: str) -> EmotionSpecResult:
    """Parse SEC output: free-form paralinguistic description.

    Also harvests an SER taxonomy hint from the summary on a best-effort
    basis so the frontend can show a small classification chip alongside
    the description (same UX as the baseline ``sec`` mode).
    """
    text = _strip_code_fence(raw).strip()

    summary = text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("summary", "description", "text", "caption", "cues"):
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    summary = val.strip()
                    break
        elif isinstance(parsed, str):
            summary = parsed.strip()
    except json.JSONDecodeError:
        pass

    label = _match_taxonomy(summary)
    return EmotionSpecResult(mode="sec", label=label, text=summary, raw_text=raw)


def parse_emotion_spec_output(
    raw_text: str, mode: EmotionSpecMode = DEFAULT_MODE
) -> EmotionSpecResult:
    """Parse the SPEC model output into a normalized :class:`EmotionSpecResult`."""
    raw = str(raw_text or "")
    if not raw.strip():
        return EmotionSpecResult(mode=mode, label="", text="", raw_text="")
    if mode == "sec":
        return _parse_sec(raw)
    return _parse_ser(raw)


async def query_emotion_spec_model(
    audio_wav_base64: str,
    *,
    mode: EmotionSpecMode = DEFAULT_MODE,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> EmotionSpecResult:
    mode = cast(EmotionSpecMode, normalize_mode(mode))
    client = get_client()
    base = (base_url or default_config.emotion_spec_vllm_base_url).rstrip("/")
    payload: dict[str, Any] = {
        "model": model_name or default_config.emotion_spec_vllm_model_name,
        "messages": _build_messages(audio_wav_base64, mode),
        "max_tokens": int(max_tokens) if max_tokens else (32 if mode == "ser" else 256),
    }
    resp = await client.post(
        f"{base}/v1/chat/completions",
        json=payload,
        timeout=timeout if timeout is not None else default_config.emotion_spec_request_timeout,
    )
    resp.raise_for_status()
    raw_text = _content_to_text(
        resp.json()["choices"][0]["message"]["content"]
    )
    return parse_emotion_spec_output(raw_text, mode=mode)
