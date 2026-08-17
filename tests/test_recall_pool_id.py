"""Unit tests for hotword_pool_id normalization/validation (change 2 slice A)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.asr.recall import normalize_hotword_pool_id  # noqa: E402


def test_blank_and_none_map_to_default_pool():
    assert normalize_hotword_pool_id(None) == ""
    assert normalize_hotword_pool_id("") == ""
    assert normalize_hotword_pool_id("   ") == ""


def test_valid_ids_are_trimmed_and_preserved():
    assert normalize_hotword_pool_id("  team-42  ") == "team-42"
    assert normalize_hotword_pool_id("a.b_c:1") == "a.b_c:1"


@pytest.mark.parametrize("bad", ["a b", "x/y", "名字", "a\tb", "p#1"])
def test_invalid_charset_raises(bad):
    with pytest.raises(ValueError):
        normalize_hotword_pool_id(bad)


def test_over_length_raises():
    with pytest.raises(ValueError):
        normalize_hotword_pool_id("x" * 129)
