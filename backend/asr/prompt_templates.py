"""Primary ASR prompt templates keyed by model training format."""

from __future__ import annotations

from typing import Any, Callable

from ..config import VALID_PRIMARY_PROMPT_TEMPLATES

PrimaryPromptBuilder = Callable[..., list[dict]]


def sanitize_hotwords(hotwords: list[str] | None) -> list[str]:
    """Drop empties, dedup, preserve order."""
    if not hotwords:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for word in hotwords:
        value = str(word or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def audio_item(wav_b64: str) -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": {"data": wav_b64, "format": "wav"},
    }


def audio_embeds_item(audio_embeds_b64: str, uuid: str) -> dict[str, Any]:
    return {
        "type": "audio_embeds",
        "audio_embeds": audio_embeds_b64,
        "uuid": uuid,
    }


def _amphion_asr(
    target_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    enrollment_wav_base64: str | None = None,
    enrollment_audio_embeds_b64: str | None = None,
    enrollment_audio_embeds_uuid: str | None = None,
    audio_embeds_b64: str | None = None,
    audio_embeds_uuid: str | None = None,
) -> list[dict]:
    """Amphion 4B swift style: text and audio interleaved in user turn."""
    _ = (
        audio_embeds_b64,
        audio_embeds_uuid,
        enrollment_audio_embeds_b64,
        enrollment_audio_embeds_uuid,
    )
    hws = sanitize_hotwords(hotwords)
    hw_str = ",".join(hws) if hws else ""
    has_enroll = bool(enrollment_wav_base64)

    content: list[dict[str, Any]] = []
    if has_enroll:
        content.append({"type": "text", "text": "Given the speaker's voice:"})
        content.append(audio_item(enrollment_wav_base64))  # type: ignore[arg-type]
        second = "\nTranscribe what this speaker says in the following audio."
        if hw_str:
            second += f"\nHotwords: {hw_str}"
        content.append({"type": "text", "text": second})
        content.append(audio_item(target_wav_base64))
    else:
        first = "Transcribe the following audio."
        if hw_str:
            first += f"\nHotwords: {hw_str}"
        content.append({"type": "text", "text": first})
        content.append(audio_item(target_wav_base64))

    return [{"role": "user", "content": content}]


def _amphion_asr_1_7b(
    target_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    enrollment_wav_base64: str | None = None,
    enrollment_audio_embeds_b64: str | None = None,
    enrollment_audio_embeds_uuid: str | None = None,
    audio_embeds_b64: str | None = None,
    audio_embeds_uuid: str | None = None,
) -> list[dict]:
    """Amphion 1.7B Qwen3-ASR style: text in system, audio-only user turn."""
    hws = sanitize_hotwords(hotwords)
    hw_str = ",".join(hws) if hws else ""
    has_enroll = bool(enrollment_wav_base64 or enrollment_audio_embeds_b64)

    system_lines: list[str] = []
    if has_enroll:
        system_lines.append("Given the speaker's voice in the first audio.")
    if hw_str:
        system_lines.append(f"Hotwords: {hw_str}")

    audio_content: list[dict[str, Any]] = []
    if has_enroll:
        if enrollment_audio_embeds_b64 and enrollment_audio_embeds_uuid:
            audio_content.append(
                audio_embeds_item(
                    enrollment_audio_embeds_b64,
                    enrollment_audio_embeds_uuid,
                )
            )
        elif enrollment_wav_base64:
            audio_content.append(audio_item(enrollment_wav_base64))
    if audio_embeds_b64 and audio_embeds_uuid and not enrollment_wav_base64:
        audio_content.append(audio_embeds_item(audio_embeds_b64, audio_embeds_uuid))
    else:
        audio_content.append(audio_item(target_wav_base64))

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": audio_content},
    ]


def _qwen3_asr(
    target_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    enrollment_wav_base64: str | None = None,
    enrollment_audio_embeds_b64: str | None = None,
    enrollment_audio_embeds_uuid: str | None = None,
    audio_embeds_b64: str | None = None,
    audio_embeds_uuid: str | None = None,
) -> list[dict]:
    """Official Qwen3-ASR format: one audio-only user turn.

    Qwen3-ASR does not support prompt hotwords, target-speaker enrollment, or
    Amphion's encoder-embedding bypass. Those values are intentionally ignored;
    hotword correction for the Qwen-only deployment happens after recognition.
    """
    _ = (
        hotwords,
        enrollment_wav_base64,
        enrollment_audio_embeds_b64,
        enrollment_audio_embeds_uuid,
        audio_embeds_b64,
        audio_embeds_uuid,
    )
    return [{"role": "user", "content": [audio_item(target_wav_base64)]}]


PRIMARY_PROMPT_BUILDERS = {
    "amphion_asr": _amphion_asr,
    "amphion_asr_1.7b": _amphion_asr_1_7b,
    "qwen3_asr": _qwen3_asr,
}

if set(PRIMARY_PROMPT_BUILDERS) != VALID_PRIMARY_PROMPT_TEMPLATES:
    raise RuntimeError(
        "PRIMARY_PROMPT_BUILDERS must match VALID_PRIMARY_PROMPT_TEMPLATES"
    )


def build_primary_messages(
    target_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    enrollment_wav_base64: str | None = None,
    enrollment_audio_embeds_b64: str | None = None,
    enrollment_audio_embeds_uuid: str | None = None,
    audio_embeds_b64: str | None = None,
    audio_embeds_uuid: str | None = None,
    template: str,
) -> list[dict]:
    """Build primary ASR chat messages for the selected model template."""
    builder = PRIMARY_PROMPT_BUILDERS.get(template)
    if builder is None:
        raise ValueError(
            f"unknown primary prompt template {template!r}; "
            f"known: {sorted(PRIMARY_PROMPT_BUILDERS)}"
        )
    return builder(
        target_wav_base64,
        hotwords=hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
        enrollment_audio_embeds_b64=enrollment_audio_embeds_b64,
        enrollment_audio_embeds_uuid=enrollment_audio_embeds_uuid,
        audio_embeds_b64=audio_embeds_b64,
        audio_embeds_uuid=audio_embeds_uuid,
    )
