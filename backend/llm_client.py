import json
import re
from pathlib import Path
from typing import Any

import httpx

from .config import (
    ASR_REQUEST_TIMEOUT,
    SECONDARY_VLLM_BASE_URL,
    SECONDARY_VLLM_MODEL_NAME,
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
)
from .prompt import EXTRACT_HOTWORD

_client: httpx.AsyncClient | None = None
_extractor_config_cache: dict[str, str] | None = None
EXTRACTED_HOTWORD_MAX_LEN = 10


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_extractor_config() -> dict[str, str]:
    """Load text-extraction model config from backend/api.json."""
    global _extractor_config_cache
    if _extractor_config_cache is not None:
        return _extractor_config_cache

    config_path = _backend_dir() / "api.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("backend/api.json is empty or invalid.")

    profile = next(iter(data.values()))
    if not isinstance(profile, dict):
        raise ValueError("backend/api.json profile format is invalid.")

    model = str(profile.get("model", "")).strip()
    api_key = str(profile.get("api_key", "")).strip()
    base_url = str(profile.get("base_url", "")).rstrip("/")
    provider = str(profile.get("provider", "")).strip() or "openai"

    if not model or not api_key or not base_url:
        raise ValueError("backend/api.json must include model, api_key, and base_url.")

    _extractor_config_cache = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }
    return _extractor_config_cache


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()
    return stripped


def _normalize_hotwords_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    raw_hotwords = payload.get("hotwords", [])
    if not isinstance(raw_hotwords, list):
        raise ValueError("`hotwords` must be a list.")
    cleaned: list[str] = []
    for item in raw_hotwords:
        if isinstance(item, str):
            value = item.strip()
            if value and value not in cleaned:
                cleaned.append(value)
    return cleaned


def _parse_hotword_json(raw_text: str) -> list[str]:
    raw = str(raw_text or "").strip()
    if not raw:
        return []
    normalized = _strip_json_fence(raw)
    try:
        return _normalize_hotwords_payload(json.loads(normalized))
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", normalized)
        if not json_match:
            raise ValueError("Could not parse JSON hotword output from model.") from None
        return _normalize_hotwords_payload(json.loads(json_match.group(0)))


def _filter_extracted_hotwords(words: list[str]) -> list[str]:
    return [word for word in words if len(word) < EXTRACTED_HOTWORD_MAX_LEN]


def _build_extract_headers(provider: str, api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _build_extract_endpoint(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


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


def build_prompt(hotwords: list[str]) -> str:
    hw_str = ",".join(hotwords) if hotwords else ""
    return f"Hotwords:{hw_str}\nTranscribe the following audio:"


def build_single_turn_messages(prompt_text: str, audio_wav_base64: str) -> list[dict]:
    # Stateless request: always send a single user turn, never append history.
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
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


def _parse_hotwords_field(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []

    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "-"}:
        return []

    return [item.strip() for item in re.split(r"[,，;；]", text) if item.strip()]


def _postprocess_asr_text(text: str) -> str:
    """Normalize provider-specific wrappers to plain transcription text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    # Qwen3-ASR may return: "language Chinese<asr_text>这边在测试东西。"
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*<asr_text>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Some responses may only include language prefix without asr tag.
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*[:：-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def parse_model_output(raw_text: str) -> dict[str, str | list[str]]:
    """Parse model output like:
    Transcription: ...
    Hotwords: ...
    Supports both real newlines and literal '\\n'.
    """
    raw = str(raw_text or "").strip()
    if not raw:
        return {"transcription": "", "reported_hotwords": [], "raw_text": ""}

    normalized = raw.replace("\\r\\n", "\n").replace("\\n", "\n")

    transcription_match = re.search(
        r"(?:^|\n)\s*transcription\s*:\s*(.+?)(?=\n\s*hotwords\s*:|\Z)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    hotwords_match = re.search(
        r"(?:^|\n)\s*hotwords\s*:\s*(.+?)(?=\n\s*[A-Za-z_]+\s*:|\Z)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )

    transcription = (
        transcription_match.group(1).strip() if transcription_match else normalized.strip()
    )
    transcription = _postprocess_asr_text(transcription)
    reported_hotwords = (
        _parse_hotwords_field(hotwords_match.group(1)) if hotwords_match else []
    )

    return {
        "transcription": transcription,
        "reported_hotwords": reported_hotwords,
        "raw_text": raw,
    }


async def _query_audio_model_by_endpoint(
    audio_wav_base64: str,
    *,
    base_url: str,
    model_name: str,
    hotwords: list[str] | None,
) -> dict[str, str | list[str]]:
    client = get_client()
    prompt_text = build_prompt(hotwords or [])
    base = base_url.rstrip("/")

    resp = await client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": build_single_turn_messages(prompt_text, audio_wav_base64),
            "max_tokens": 512,
        },
        timeout=ASR_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    return parse_model_output(raw_text)


async def query_audio_model(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
) -> dict[str, str | list[str]]:
    return await _query_audio_model_by_endpoint(
        audio_wav_base64,
        base_url=VLLM_BASE_URL,
        model_name=VLLM_MODEL_NAME,
        hotwords=hotwords,
    )


async def query_audio_model_secondary(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
) -> dict[str, str | list[str]]:
    return await _query_audio_model_by_endpoint(
        audio_wav_base64,
        base_url=SECONDARY_VLLM_BASE_URL,
        model_name=SECONDARY_VLLM_MODEL_NAME,
        hotwords=hotwords,
    )


async def query_text_hotwords(text: str) -> list[str]:
    """Extract hotwords from long text using the model config in backend/api.json."""
    source = str(text or "").strip()
    if not source:
        return []

    client = get_client()
    cfg = _load_extractor_config()
    endpoint = _build_extract_endpoint(cfg["base_url"])
    headers = _build_extract_headers(cfg["provider"], cfg["api_key"])

    resp = await client.post(
        endpoint,
        headers=headers,
        json={
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": EXTRACT_HOTWORD},
                {"role": "user", "content": source},
            ],
            "max_tokens": 512,
        },
    )
    resp.raise_for_status()
    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    return _filter_extracted_hotwords(_parse_hotword_json(raw_text))


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
