#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONTRACTS = {
    "qwen3-asr-1.7b": {
        "architecture": "Qwen3ASRForConditionalGeneration",
        "required": {
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer_config.json",
        },
    },
    "amphion-spec": {
        "architecture": "AmphionASRForConditionalGeneration",
        "required": {
            "config.json",
            "model.safetensors",
            "configuration_amphion_asr.py",
            "modeling_amphion_asr.py",
            "processing_amphion_asr.py",
            "tokenizer_config.json",
        },
    },
}


def main() -> None:
    for name, contract in MODEL_CONTRACTS.items():
        model_dir = PROJECT_ROOT / "models" / name
        missing = sorted(
            filename for filename in contract["required"] if not (model_dir / filename).is_file()
        )
        if missing:
            raise SystemExit(f"{name}: missing required files: {', '.join(missing)}")
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        architectures = config.get("architectures") or []
        if contract["architecture"] not in architectures:
            raise SystemExit(
                f"{name}: expected architecture {contract['architecture']}, got {architectures}"
            )
        weight_bytes = sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
        if not weight_bytes:
            raise SystemExit(f"{name}: no safetensors weights found")
        print(f"{name}: ok ({weight_bytes / 1024**3:.2f} GiB weights)")


if __name__ == "__main__":
    main()
