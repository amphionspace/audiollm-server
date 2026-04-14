import difflib
import re
import unicodedata
from typing import Any, TypedDict

from ..config import (
    FUSION_DISAGREEMENT_THRESHOLD,
    FUSION_HOTWORD_BOOST,
    FUSION_MAX_REPETITION_RATIO,
    FUSION_MIN_PRIMARY_SCORE,
    FUSION_PRIMARY_SCORE_MARGIN,
    FUSION_SIMILARITY_THRESHOLD,
)
from .client import ASRResult


class FusionMeta(TypedDict):
    selected: str
    reason: str
    similarity: float | None
    threshold: float | None
    disagreement: float | None
    scores: dict[str, float]
    metrics: dict[str, Any] | None
    normalized_preview: dict[str, str]


class FusionResult(TypedDict):
    text: str
    model_hotwords: list[str]
    primary_text: str
    secondary_text: str
    fusion: FusionMeta


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """NFKC-normalize, lower-case, harmonize punctuation, strip fillers."""
    raw = str(text or "")
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFKC", raw).strip().lower()
    normalized = normalized.replace("，", ",").replace("。", ".").replace("；", ";")
    normalized = normalized.replace("：", ":").replace("？", "?").replace("！", "!")
    normalized = re.sub(r"[`~^*_=+|\\]+", " ", normalized)
    normalized = re.sub(r"\b(um+|uh+|emm+)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [token for token in re.split(r"[\s,.;:!?]+", normalized) if token]


def _repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    unique_tokens = len(set(tokens))
    repeated = max(0, len(tokens) - unique_tokens)
    return repeated / len(tokens)


def _longest_run_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    max_run = 1
    current_run = 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run / len(tokens)


def _abnormal_char_ratio(text: str) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 1.0
    allowed = 0
    for ch in normalized:
        if ch.isalnum() or ch in {" ", ",", ".", ";", ":", "!", "?", "'", '"', "-", "/"}:
            allowed += 1
    return 1.0 - (allowed / len(normalized))


def _hotword_hit_count(text: str, hotwords: list[str]) -> int:
    normalized_text = normalize_text(text)
    hits = 0
    for hotword in hotwords:
        hw_norm = normalize_text(hotword)
        if hw_norm and hw_norm in normalized_text:
            hits += 1
    return hits


def _filter_reported_hotwords(
    reported: list[str], original: list[str]
) -> list[str]:
    """Keep only reported hotwords that exactly match an original hotword.

    ASR models sometimes report homophones (e.g. 执音 vs 挚音) in their
    Hotwords output field.  Passing those through to the frontend would cause
    incorrect highlighting, so we intersect with the canonical list.
    """
    orig_set = {normalize_text(w) for w in original if w}
    return [w for w in reported if normalize_text(w) in orig_set]


def _quality_score(text: str, hotwords: list[str], hotword_boost: float) -> dict[str, float]:
    normalized = normalize_text(text)
    if not normalized:
        return {
            "score": 0.0,
            "repetition_ratio": 1.0,
            "longest_run_ratio": 1.0,
            "abnormal_char_ratio": 1.0,
            "hotword_hits": 0.0,
        }

    tokens = _tokenize(normalized)
    repetition = _repetition_ratio(tokens)
    longest_run = _longest_run_ratio(tokens)
    abnormal_char = _abnormal_char_ratio(normalized)
    hotword_hits = _hotword_hit_count(normalized, hotwords)

    score = 1.0
    score -= min(0.45, repetition * 0.7)
    score -= min(0.35, longest_run * 0.6)
    score -= min(0.30, abnormal_char * 0.5)
    score += min(0.35, hotword_hits * hotword_boost)
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "repetition_ratio": round(repetition, 4),
        "longest_run_ratio": round(longest_run, 4),
        "abnormal_char_ratio": round(abnormal_char, 4),
        "hotword_hits": float(hotword_hits),
    }


def _text_similarity(a: str, b: str) -> float:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


# ---------------------------------------------------------------------------
# Dual-ASR fusion
# ---------------------------------------------------------------------------

def choose_fused_result(
    primary_result: ASRResult | None,
    secondary_result: ASRResult | None,
    hotwords: list[str],
) -> FusionResult:
    primary_text = str((primary_result or {}).get("transcription") or "")
    secondary_text = str((secondary_result or {}).get("transcription") or "")
    primary_hotwords = _filter_reported_hotwords(
        list((primary_result or {}).get("reported_hotwords") or []), hotwords
    )
    secondary_hotwords = _filter_reported_hotwords(
        list((secondary_result or {}).get("reported_hotwords") or []), hotwords
    )
    normalized_primary = normalize_text(primary_text)
    normalized_secondary = normalize_text(secondary_text)

    def _meta(
        selected: str,
        reason: str,
        *,
        similarity: float | None = None,
        threshold: float | None = None,
        disagreement: float | None = None,
        scores: dict[str, float] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> FusionMeta:
        return FusionMeta(
            selected=selected,
            reason=reason,
            similarity=similarity,
            threshold=threshold,
            disagreement=disagreement,
            scores=scores or {"primary": 0.0, "secondary": 0.0},
            metrics=metrics,
            normalized_preview={
                "primary": normalized_primary,
                "secondary": normalized_secondary,
            },
        )

    if not secondary_text:
        return FusionResult(
            text="",
            model_hotwords=[],
            primary_text=primary_text,
            secondary_text=secondary_text,
            fusion=_meta("silence", "secondary_empty_force_silence"),
        )

    if secondary_text and not primary_text:
        return FusionResult(
            text=secondary_text,
            model_hotwords=secondary_hotwords,
            primary_text=primary_text,
            secondary_text=secondary_text,
            fusion=_meta("secondary_only", "primary_empty",
                         scores={"primary": 0.0, "secondary": 1.0}),
        )

    if not primary_text and not secondary_text:
        return FusionResult(
            text="",
            model_hotwords=[],
            primary_text="",
            secondary_text="",
            fusion=_meta("empty", "both_empty"),
        )

    similarity = _text_similarity(primary_text, secondary_text)
    primary_metrics = _quality_score(primary_text, hotwords, FUSION_HOTWORD_BOOST)
    secondary_metrics = _quality_score(secondary_text, hotwords, 0.0)
    primary_score = float(primary_metrics["score"])
    secondary_score = float(secondary_metrics["score"])
    primary_repetition = float(primary_metrics["repetition_ratio"])
    primary_hotword_hits = int(primary_metrics["hotword_hits"])
    disagreement = 1.0 - similarity

    primary_is_hallucination_risk = (
        primary_repetition > FUSION_MAX_REPETITION_RATIO
        or disagreement > FUSION_DISAGREEMENT_THRESHOLD
        and primary_metrics["hotword_hits"] <= secondary_metrics["hotword_hits"]
    )
    primary_meets_bar = primary_score >= FUSION_MIN_PRIMARY_SCORE
    primary_better = primary_score >= (secondary_score + FUSION_PRIMARY_SCORE_MARGIN)

    if primary_hotword_hits > 0:
        selected = "primary_hotword_hit"
        reason = "primary_hits_hotword"
        selected_text = primary_text
        selected_hotwords = primary_hotwords
    elif primary_is_hallucination_risk and secondary_text:
        selected = "secondary_qwen_fallback"
        reason = "primary_hallucination_risk"
        selected_text = secondary_text
        selected_hotwords = secondary_hotwords
    elif similarity >= FUSION_SIMILARITY_THRESHOLD and primary_meets_bar:
        selected = "primary_agreement"
        reason = "high_similarity_and_primary_valid"
        selected_text = primary_text
        selected_hotwords = primary_hotwords
    elif primary_better and primary_meets_bar:
        selected = "primary_hotword_advantage"
        reason = "primary_score_margin"
        selected_text = primary_text
        selected_hotwords = primary_hotwords
    else:
        selected = "secondary_qwen_fallback"
        reason = "primary_not_confident"
        selected_text = secondary_text
        selected_hotwords = secondary_hotwords

    return FusionResult(
        text=selected_text,
        model_hotwords=selected_hotwords,
        primary_text=primary_text,
        secondary_text=secondary_text,
        fusion=_meta(
            selected,
            reason,
            similarity=round(similarity, 4),
            threshold=FUSION_SIMILARITY_THRESHOLD,
            disagreement=round(disagreement, 4),
            scores={
                "primary": round(primary_score, 4),
                "secondary": round(secondary_score, 4),
            },
            metrics={
                "primary": primary_metrics,
                "secondary": secondary_metrics,
            },
        ),
    )
