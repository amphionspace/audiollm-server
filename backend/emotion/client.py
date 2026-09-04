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
import math
import re
from typing import Any, TypedDict

from ..config import default_config, get_service_upstream
from ..http_client import get_client
from .prompt import (
    DEFAULT_MODE,
    SER_TAXONOMY,
    EmotionMode,
    get_prompt,
    normalize_mode,
)

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


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
    top_emotions: list[dict[str, object]]
    best_label: str
    best_score: float


_LABEL_LOOKUP: dict[str, str] = {label.casefold(): label for label in SER_TAXONOMY}


def _taxonomy_label_for_token(raw: object) -> str:
    token = str(raw or "").strip().casefold()
    if not token:
        return ""
    matches = [
        label
        for label in SER_TAXONOMY
        if (head := label.split("/", 1)[0].casefold()) == token
        or head.startswith(token)
    ]
    return matches[0] if len(matches) == 1 else ""


def _rank_top_emotions(choice: object) -> list[dict[str, object]]:
    """Normalize SER first-token logprobs into taxonomy posterior scores."""
    if not isinstance(choice, dict):
        return []
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return []
    candidates = content[0].get("top_logprobs")
    if not isinstance(candidates, list):
        return []
    by_label: dict[str, float] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        label = _taxonomy_label_for_token(candidate.get("token"))
        value = candidate.get("logprob")
        if not label or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        by_label[label] = max(by_label.get(label, -math.inf), float(value))
    if not by_label:
        return []
    maximum = max(by_label.values())
    denominator = sum(math.exp(value - maximum) for value in by_label.values())
    ranked = [
        {"label": label, "score": round(math.exp(value - maximum) / denominator, 6)}
        for label, value in by_label.items()
    ]
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked


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


def _build_messages(
    audio_wav_base64: str,
    mode: EmotionMode,
    language: str = "",
) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": get_prompt(mode, language)},
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


def _needs_sec_language_translation(text: str, language: str) -> bool:
    normalized = str(language or "").strip().lower()
    has_cjk = bool(_CJK_RE.search(text))
    if normalized in {"zh", "zh-cn", "cn", "chinese"}:
        return bool(text.strip()) and (not has_cjk or bool(_LATIN_RE.search(text)))
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return has_cjk
    return False


async def _translate_sec_output(text: str, language: str) -> str:
    upstream = get_service_upstream("speech_refine")
    if upstream is None or not upstream.api_key:
        raise RuntimeError("speech_refine upstream is not configured")
    normalized = str(language or "").strip().lower()
    target = (
        "Simplified Chinese"
        if normalized in {"zh", "zh-cn", "cn", "chinese"}
        else "English"
    )
    response = await get_client().post(
        f"{upstream.base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {upstream.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": upstream.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Translate the emotion description into {target}. Preserve its meaning "
                        "and intensity. Translate every word, including technical emotion terms. "
                        "Return only the translated description and do not mix languages."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "max_tokens": upstream.max_tokens or 512,
            "thinking": {"type": "disabled"},
        },
        timeout=upstream.timeout,
    )
    response.raise_for_status()
    translated = _content_to_text(response.json()["choices"][0]["message"]["content"])
    if not translated.strip() or _needs_sec_language_translation(translated, language):
        raise RuntimeError(f"SEC translation did not produce {target}")
    return translated.strip()


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

    return EmotionResult(
        mode="ser", label=label, text=label, raw_text=raw,
        top_emotions=[], best_label="", best_score=0.0,
    )


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
    return EmotionResult(
        mode="sec", label=label, text=summary, raw_text=raw,
        top_emotions=[], best_label="", best_score=0.0,
    )


def parse_emotion_output(raw_text: str, mode: EmotionMode = DEFAULT_MODE) -> EmotionResult:
    """Parse the model output into a normalized :class:`EmotionResult`."""
    raw = str(raw_text or "")
    if not raw.strip():
        return EmotionResult(
            mode=mode, label="", text="", raw_text="",
            top_emotions=[], best_label="", best_score=0.0,
        )
    if mode == "sec":
        return _parse_sec(raw)
    return _parse_ser(raw)


async def query_emotion_model(
    audio_wav_base64: str,
    *,
    mode: EmotionMode = DEFAULT_MODE,
    language: str = "",
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> EmotionResult:
    mode = normalize_mode(mode)
    client = get_client()
    base = (base_url or default_config.emotion_vllm_base_url).rstrip("/")
    payload: dict[str, Any] = {
        "model": model_name or default_config.emotion_vllm_model_name,
        "messages": _build_messages(audio_wav_base64, mode, language),
        "max_tokens": int(max_tokens) if max_tokens else (32 if mode == "ser" else 256),
    }
    if mode == "ser":
        payload.update({"logprobs": True, "top_logprobs": 20})
    resp = await client.post(
        f"{base}/v1/chat/completions",
        json=payload,
        timeout=timeout if timeout is not None else default_config.emotion_request_timeout,
    )
    resp.raise_for_status()
    response_payload = resp.json()
    choice = response_payload["choices"][0]
    raw_text = _content_to_text(choice["message"]["content"])
    result = parse_emotion_output(raw_text, mode=mode)
    if mode == "sec" and _needs_sec_language_translation(result["text"], language):
        result["text"] = await _translate_sec_output(result["text"], language)
    if mode == "ser":
        ranked = _rank_top_emotions(choice)
        result["top_emotions"] = ranked[:3]
        result["best_label"] = result["label"]
        for candidate in ranked:
            if candidate["label"] == result["label"]:
                result["best_score"] = float(candidate["score"])
                break
    return result
