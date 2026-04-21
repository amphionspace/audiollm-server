"""vLLM (OpenAI-compatible) client for the AmphionASR SER/SEC model.

Mirrors the surface area of :mod:`backend.asr.client` so the engine can call
this without a special case.

The Amphion model is trained to emit:

- ``ser`` mode: a single label string from :data:`SER_TAXONOMY`
  (e.g. ``"Happy"``, ``"Sad"`` ...).
- ``sec`` mode: a free-form natural-language emotion summary.

Parsing therefore biases towards plain text. JSON wrapping (which a
post-trained model might still emit) is tolerated as a best-effort fallback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from ..config import (
    EMOTION_REQUEST_TIMEOUT,
    EMOTION_VLLM_BASE_URL,
    EMOTION_VLLM_MODEL_NAME,
)
from ..http_client import get_client
from .prompt import (
    DEFAULT_MODE,
    SER_TAXONOMY,
    EmotionMode,
    get_prompt,
    normalize_mode,
)

logger = logging.getLogger(__name__)


class EmotionResult(TypedDict):
    """Normalized emotion-model output.

    Field semantics depend on ``mode``:

    - ``mode == "ser"``: ``label`` holds one of :data:`SER_TAXONOMY`
      (or ``""`` if parsing failed); ``text`` mirrors the raw label for
      convenience.
    - ``mode == "sec"``: ``text`` holds the free-form summary; ``label``
      is the best-effort taxonomy hit found inside that summary (may be
      ``""``).
    """

    mode: EmotionMode
    label: str
    text: str
    raw_text: str


_LABEL_LOOKUP: dict[str, str] = {label.casefold(): label for label in SER_TAXONOMY}


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


def _build_messages(audio_wav_base64: str, mode: EmotionMode) -> list[dict]:
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


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def _match_taxonomy(text: str) -> str:
    """Find the first SER taxonomy label appearing in *text* (case-insensitive)."""
    if not text:
        return ""
    lowered = text.casefold()
    exact = lowered.strip().rstrip(".,!?;:\"' \n\t")
    if exact in _LABEL_LOOKUP:
        return _LABEL_LOOKUP[exact]
    for canonical_lower, canonical in _LABEL_LOOKUP.items():
        head = canonical_lower.split("/", 1)[0]
        pattern = rf"\b{re.escape(head)}\b"
        if re.search(pattern, lowered):
            return canonical
    return ""


def _parse_ser(raw: str) -> EmotionResult:
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
        logger.warning("Could not map SER output to taxonomy: %.200s", raw)

    return EmotionResult(mode="ser", label=label, text=label, raw_text=raw)


def _parse_sec(raw: str) -> EmotionResult:
    """Parse SEC output: free-form description; also harvest a label hint."""
    text = _strip_code_fence(raw).strip()

    summary = text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("summary", "description", "text", "caption"):
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    summary = val.strip()
                    break
        elif isinstance(parsed, str):
            summary = parsed.strip()
    except json.JSONDecodeError:
        pass

    label = _match_taxonomy(summary)
    return EmotionResult(mode="sec", label=label, text=summary, raw_text=raw)


def parse_emotion_output(raw_text: str, mode: EmotionMode = DEFAULT_MODE) -> EmotionResult:
    """Parse the model output into a normalized :class:`EmotionResult`."""
    raw = str(raw_text or "")
    if not raw.strip():
        return EmotionResult(mode=mode, label="", text="", raw_text="")
    if mode == "sec":
        return _parse_sec(raw)
    return _parse_ser(raw)


async def query_emotion_model(
    audio_wav_base64: str,
    *,
    mode: EmotionMode = DEFAULT_MODE,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> EmotionResult:
    mode = normalize_mode(mode)
    client = get_client()
    base = (base_url or EMOTION_VLLM_BASE_URL).rstrip("/")
    payload: dict[str, Any] = {
        "model": model_name or EMOTION_VLLM_MODEL_NAME,
        "messages": _build_messages(audio_wav_base64, mode),
        "max_tokens": int(max_tokens) if max_tokens else (32 if mode == "ser" else 256),
    }
    resp = await client.post(
        f"{base}/v1/chat/completions",
        json=payload,
        timeout=timeout if timeout is not None else EMOTION_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    raw_text = _content_to_text(
        resp.json()["choices"][0]["message"]["content"]
    )
    return parse_emotion_output(raw_text, mode=mode)
