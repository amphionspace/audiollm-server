from __future__ import annotations

import json
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("acl", types.ModuleType("acl"))
sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))

if "backend.asr" not in sys.modules:
    asr_pkg = types.ModuleType("backend.asr")
    asr_pkg.__path__ = [str(ROOT / "backend/asr")]
    sys.modules["backend.asr"] = asr_pkg

spec = importlib.util.spec_from_file_location(
    "backend.asr.ctc_streaming",
    ROOT / "backend/asr/ctc_streaming.py",
)
assert spec is not None and spec.loader is not None
ctc_streaming = importlib.util.module_from_spec(spec)
sys.modules["backend.asr.ctc_streaming"] = ctc_streaming
spec.loader.exec_module(ctc_streaming)
_decode_tokens = ctc_streaming._decode_tokens
_load_tokens = ctc_streaming._load_tokens


def test_sentencepiece_byte_fallback_tokens_decode_utf8(tmp_path: Path) -> None:
    tokens_path = tmp_path / "tokens.txt"
    tokens_path.write_text(
        "\n".join(
            [
                "<blk> 0",
                "<sos> 1",
                "<eos> 2",
                "<pad> 3",
                "<unk> 4",
                "<0xE4> 5",
                "<0xBD> 6",
                "<0xA0> 7",
                "▁hello 8",
                "▁world 9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, mode = _load_tokens(str(tokens_path))
    assert mode == "sentencepiece_byte_fallback_bpe"
    assert _decode_tokens(str(tokens_path), [8, 9, 5, 6, 7]) == "hello world你"


def test_pkufool_candidate_contract_reports_token_vocab_match() -> None:
    candidate_dir = (
        ROOT.parent / "models/ctc_candidates/pkufool-zipformer-small-streaming-ctc-zhen"
    )
    if not candidate_dir.exists():
        candidate_dir = Path(
            "/home/workspace/models/ctc_candidates/pkufool-zipformer-small-streaming-ctc-zhen"
        )

    meta = json.loads((candidate_dir / "candidate_meta.json").read_text(encoding="utf-8"))
    contract = json.loads(
        (candidate_dir / "onnx_contract_report.json").read_text(encoding="utf-8")
    )
    tokenizer = json.loads((candidate_dir / "tokenizer_report.json").read_text(encoding="utf-8"))

    assert meta["selected_candidate"]["parameter_count"] == 25325984
    assert meta["tokenizer"]["detected_mode"] == "sentencepiece_byte_fallback_bpe"
    assert tokenizer["byte_fallback_count"] == 256
    assert contract["outputs"][0]["name"] == "log_probs"
    assert contract["outputs"][0]["shape"][-1] == meta["tokenizer"]["tokens_count"] == 11661
