#!/usr/bin/env python3
"""Build a deterministic OSCAR MLA calibration manifest."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from vllm.model_executor.layers.quantization.oscar_mla.manifest import (
    CalibrationSource,
    build_calibration_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    result = build_calibration_manifest(
        sources=[CalibrationSource.from_dict(source) for source in config["sources"]],
        official_manifest_path=args.official_manifest,
        quotas=config["quotas"],
        tokenizer=tokenizer,
        tokenizer_sha256=config["tokenizer_sha256"],
        seed=config["seed"],
        holdout_fraction=config["holdout_fraction"],
        max_chunk_tokens=config["max_chunk_tokens"],
        output_path=args.output,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
