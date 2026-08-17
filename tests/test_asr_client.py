from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.asr.client import _merge_recalled_and_request_hotwords  # noqa: E402


def test_merge_recalled_and_request_hotwords_prefers_session_words_first() -> None:
    # Session/request hotwords rank first (contract V0.4-review 会话热词优先);
    # recalled pool words follow, and a recalled word that duplicates a request
    # word is dropped.
    merged = _merge_recalled_and_request_hotwords(
        ["警务通", "张三"],
        ["张三", "鼎桥", "  现场词  "],
    )

    assert merged == ["张三", "鼎桥", "现场词", "警务通"]


def test_merge_recalled_and_request_hotwords_uses_request_when_recall_empty() -> None:
    assert _merge_recalled_and_request_hotwords([], ["鼎桥"]) == ["鼎桥"]


def test_merge_suppresses_homophone_recall_in_favor_of_session_hotword() -> None:
    # 王惠 (session) and 王慧 (recalled) share the pinyin key wang-hui; the
    # recalled homophone must be suppressed so the session hotword wins.
    merged = _merge_recalled_and_request_hotwords(
        ["王慧", "鼎桥"],
        ["王惠"],
    )

    assert "王惠" in merged
    assert "王慧" not in merged
    assert merged.index("王惠") == 0
    assert "鼎桥" in merged
