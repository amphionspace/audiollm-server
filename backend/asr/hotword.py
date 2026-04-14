import json
import logging
import re
from pathlib import Path
from typing import Any

from ..http_client import get_client
from .client import _content_to_text
from .prompt import EXTRACT_HOTWORD

logger = logging.getLogger(__name__)

EXTRACTED_HOTWORD_MAX_LEN = 10

_extractor_config_cache: dict[str, str] | None = None


def sanitize_hotwords(words: Any) -> list[str]:
    """Deduplicate a hotword list."""
    if not isinstance(words, list):
        return []
    cleaned: list[str] = []
    for item in words:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in cleaned:
            continue
        cleaned.append(value)
    return cleaned


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent


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

    # 1) Direct JSON object parse
    try:
        return _normalize_hotwords_payload(json.loads(normalized))
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) Extract first {...} block
    json_match = re.search(r"\{[\s\S]*\}", normalized)
    if json_match:
        try:
            return _normalize_hotwords_payload(json.loads(json_match.group(0)))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3) Bare JSON array: ["term1", "term2"]
    arr_match = re.search(r"\[[\s\S]*\]", normalized)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group(0))
            if isinstance(parsed, list):
                return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass

    # 4) Comma / newline separated plain text
    items = re.split(r"[,，\n;；]+", normalized)
    plain = [s.strip().strip('"\'') for s in items]
    plain = [s for s in plain if s and not s.startswith("{")]
    if plain:
        logger.warning("Hotword extraction fell back to plain-text parse: %s", plain)
        return list(dict.fromkeys(plain))

    logger.warning("Could not parse hotword output, returning empty: %.200s", raw)
    return []


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
