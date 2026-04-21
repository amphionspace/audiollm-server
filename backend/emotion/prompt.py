"""Prompt templates for the audio emotion model (aligned with AmphionASR).

Reference (training-time prompts in ``AmphionASR/src/train.py``)::

    TASK_PROMPTS = {
        "ser": "Classify the emotion of the following audio:{speech}",
        "sec": "Describe the emotion of the following audio:{speech}",
    }

When serving via vLLM (OpenAI-compatible chat completions) the ``{speech}``
placeholder is replaced by an ``input_audio`` content item, so on the wire we
only need the prompt prefix as plain text.

The 8-way SER taxonomy mirrors ``AmphionASR/local/prepare_ser_manifests.py``.
The model is trained to emit exactly one of these label strings (preserving
the original capitalization) for the ``ser`` task; for the ``sec`` task it
emits a free-form natural-language emotion summary.
"""

from __future__ import annotations

from typing import Literal

EmotionMode = Literal["ser", "sec"]

SER_TAXONOMY: tuple[str, ...] = (
    "Neutral",
    "Happy",
    "Sad",
    "Angry",
    "Fear",
    "Disgust",
    "Surprise",
    "Other/Complex",
)

SER_PROMPT = "Classify the emotion of the following audio:"
SEC_PROMPT = "Describe the emotion of the following audio:"

PROMPTS: dict[str, str] = {
    "ser": SER_PROMPT,
    "sec": SEC_PROMPT,
}

DEFAULT_MODE: EmotionMode = "ser"


def normalize_mode(value: object) -> EmotionMode:
    """Coerce a user-supplied mode string to one of the supported modes."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in PROMPTS:
            return lowered  # type: ignore[return-value]
    return DEFAULT_MODE


def get_prompt(mode: EmotionMode) -> str:
    return PROMPTS[mode]
