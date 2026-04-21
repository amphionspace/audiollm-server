"""Prompt builder for Target-Speaker ASR (TS-ASR).

The prompt format mirrors AmphionASR's ms-swift SFT recipe (see
``AmphionASR/src/integrations/vllm/test_vllm_inference.py``)::

    Given the speaker's voice:<audio_enroll>
    Transcribe what this speaker says in the following audio:<audio_mixed>

This module intentionally exposes a *builder function* rather than a static
string template: TS-ASR is still evolving (optional hotwords, speaker voice
traits, styling instructions, ...), and future variants should be added here
without touching the task engine or the client.
"""

from __future__ import annotations

from typing import Any

ENROLL_PREFIX = "Given the speaker's voice:"
TRANSCRIBE_PREFIX = "Transcribe what this speaker says in the following audio:"


def _audio_chunk(wav_base64: str) -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": {"data": wav_base64, "format": "wav"},
    }


def format_hotwords_segment(hotwords: list[str] | None) -> str:
    """Format a hotwords sidecar text segment.

    Returns an empty string when the list is empty / None so callers can
    unconditionally concatenate the result. The segment is meant to be
    inserted between the enrollment audio and the transcribe instruction.
    """
    if not hotwords:
        return ""
    cleaned = [str(h).strip() for h in hotwords if str(h or "").strip()]
    if not cleaned:
        return ""
    return "\nHotwords: " + ",".join(cleaned) + "."


def format_voice_traits_segment(voice_traits: str | None) -> str:
    """Format an optional speaker-trait description segment."""
    if not voice_traits:
        return ""
    trimmed = str(voice_traits).strip()
    if not trimmed:
        return ""
    # Normalize trailing punctuation to keep the prompt stable.
    if trimmed.endswith(("." , "!", "?")):
        return "\nSpeaker traits: " + trimmed
    return "\nSpeaker traits: " + trimmed + "."


def build_tsasr_content(
    enrollment_wav_b64: str,
    mixed_wav_b64: str,
    *,
    hotwords: list[str] | None = None,
    voice_traits: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the OpenAI-compatible chat ``content`` list for TS-ASR.

    Output layout (positions in ``content``)::

        [0]    text:  ENROLL_PREFIX
        [1]    audio: enrollment
        [...]  optional text sidecars (voice traits, hotwords)
        [-2]   text:  "\\n" + TRANSCRIBE_PREFIX
        [-1]   audio: mixed

    The enrollment + instruction ordering is fixed (aligned with the
    training-time template). Optional sidecars are appended after the
    enrollment audio so that they apply to the following transcribe step.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": ENROLL_PREFIX},
        _audio_chunk(enrollment_wav_b64),
    ]

    traits_segment = format_voice_traits_segment(voice_traits)
    if traits_segment:
        content.append({"type": "text", "text": traits_segment})

    hotwords_segment = format_hotwords_segment(hotwords)
    if hotwords_segment:
        content.append({"type": "text", "text": hotwords_segment})

    content.append({"type": "text", "text": "\n" + TRANSCRIBE_PREFIX})
    content.append(_audio_chunk(mixed_wav_b64))
    return content
