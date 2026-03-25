import httpx

from .config import VLLM_BASE_URL, VLLM_MODEL_NAME

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


def build_prompt(hotwords: list[str]) -> str:
    hw_str = ",".join(hotwords) if hotwords else ""
    return f"Hotwords:{hw_str}\nTranscribe the following audio:"


async def query_audio_model(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
) -> str:
    client = get_client()
    prompt_text = build_prompt(hotwords or [])

    resp = await client.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json={
            "model": VLLM_MODEL_NAME,
            "messages": [
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
            ],
            "max_tokens": 512,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
