"""Prompt templates for the AmphionSPEC paralinguistic-emotion model.

The model is trained with two prompts (literal strings — do NOT rephrase):

- ``ser``  : ``Classify the emotion of the following audio:{speech}``
  Identical to the baseline emotion model; emits one of the 8-way
  :data:`backend.emotion.prompt.SER_TAXONOMY` labels.
- ``sec``  : ``Describe the paralinguistic emotion cues of the following audio:{speech}``
  Free-form description of paralinguistic cues (prosody, tempo, voice
  quality, etc.). ``sec`` is the public protocol name; AmphionSPEC remains
  an internal model identity only.

When serving via vLLM (OpenAI-compatible chat completions) the ``{speech}``
placeholder is replaced by an ``input_audio`` content item, so on the wire
we only need the prompt prefix as plain text.
"""

from __future__ import annotations

from typing import Literal

EmotionSpecMode = Literal["ser", "sec"]

SER_PROMPT = "Classify the emotion of the following audio:"
SEC_PROMPT = "Describe the paralinguistic emotion cues of the following audio:"

PROMPTS: dict[str, str] = {
    "ser": SER_PROMPT,
    "sec": SEC_PROMPT,
}

DEFAULT_MODE: EmotionSpecMode = "sec"


def normalize_mode(value: object) -> EmotionSpecMode:
    """Validate the public emotion mode; only ``ser`` and ``sec`` exist."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in PROMPTS:
            return lowered  # type: ignore[return-value]
        if not lowered:
            return DEFAULT_MODE
    raise ValueError("mode must be ser or sec")


def get_prompt(mode: EmotionSpecMode) -> str:
    return PROMPTS[mode]
