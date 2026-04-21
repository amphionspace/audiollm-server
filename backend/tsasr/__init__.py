"""Target-Speaker ASR (TS-ASR) domain logic.

A deliberately-isolated subpackage because TS-ASR is a short-term demo on top
of the existing Amphion vLLM endpoint: the prompt template, the model, and
the optional knobs (hotwords / voice traits / ...) are all expected to
evolve. Keeping the prompt builder, client, and enrollment decoder here
means that the engine shell in ``backend/tasks/ts_asr.py`` stays thin and
the blast radius of future prompt / model changes is one directory.

Not intended to be imported by the standard ASR pipeline.
"""

from __future__ import annotations

from .client import TsAsrResult, query_tsasr_model
from .enrollment import EnrollmentError, decode_enrollment
from .prompt import (
    ENROLL_PREFIX,
    TRANSCRIBE_PREFIX,
    build_tsasr_content,
    format_hotwords_segment,
    format_voice_traits_segment,
)

__all__ = [
    "ENROLL_PREFIX",
    "EnrollmentError",
    "TRANSCRIBE_PREFIX",
    "TsAsrResult",
    "build_tsasr_content",
    "decode_enrollment",
    "format_hotwords_segment",
    "format_voice_traits_segment",
    "query_tsasr_model",
]
