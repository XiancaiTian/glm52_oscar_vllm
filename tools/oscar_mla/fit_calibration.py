#!/usr/bin/env python3
"""Fit shared OSCAR MLA rotations from complete TP capture payloads."""

import argparse
import json
import os
from pathlib import Path

from vllm.model_executor.layers.quantization.oscar_mla.artifact import (
    ArtifactMetadata,
    write_rotation_artifact,
)
from vllm.model_executor.layers.quantization.oscar_mla.fit import (
    load_and_merge_capture_layer,
    search_shared_rotations,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-capture-dir", type=Path, required=True)
    parser.add_argument("--holdout-capture-dir", type=Path, required=True)
    parser.add_argument("--calibration-code-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    num_layers = int(config["num_layers"])
    layer_template = config["layer_name_template"]
    common = {
        "tp_size": int(config["tp_size"]),
        "latent_rank": int(config["latent_rank"]),
    }
    train_layers = {}
    holdout_layers = {}
    for layer in range(num_layers):
        layer_name = layer_template.format(layer=layer)
        train_layers[layer] = load_and_merge_capture_layer(
            args.train_capture_dir,
            layer_name,
            split="train",
            token_budget=int(config["train_token_budget"]),
            **common,
        )
        holdout_layers[layer] = load_and_merge_capture_layer(
            args.holdout_capture_dir,
            layer_name,
            split="holdout",
            token_budget=int(config["holdout_token_budget"]),
            **common,
        )

    alpha_grid = tuple(float(value) for value in config["alpha_grid"])
    clip_grid = tuple(float(value) for value in config["clip_grid"])
    result = search_shared_rotations(
        train_layers,
        holdout_layers,
        group_size=int(config["group_size"]),
        alpha_grid=alpha_grid,
        clip_grid=clip_grid,
    )
    summary = {
        "format_version": 1,
        "alpha": result.alpha,
        "alpha_grid": alpha_grid,
        "alpha_losses": result.alpha_losses,
        "normalized_loss": result.normalized_loss,
        "clip_grid": clip_grid,
        "layers": {
            str(layer): {
                "clip_ratio": layer_result.clip_ratio,
                "normalized_loss": layer_result.normalized_loss,
                "train_score_samples": train_layers[layer].score_samples,
                "train_value_samples": train_layers[layer].value_samples,
                "train_latent_samples": train_layers[layer].latent_samples_count,
                "holdout_score_samples": holdout_layers[layer].score_samples,
                "holdout_value_samples": holdout_layers[layer].value_samples,
                "holdout_latent_samples": holdout_layers[layer].latent_samples_count,
                "holdout_reservoir_rows": holdout_layers[layer].latent_samples.shape[0],
            }
            for layer, layer_result in result.layers.items()
        },
    }
    _atomic_json(args.output_dir / "search_summary.json", summary)

    metadata = ArtifactMetadata(
        model_config_sha256=config["model_config_sha256"],
        checkpoint_manifest_sha256=config["checkpoint_manifest_sha256"],
        expert_mapping_sha256=config["expert_mapping_sha256"],
        calibration_code_commit=args.calibration_code_commit,
        calibration_manifest_sha256=config["calibration_manifest_sha256"],
        seed=int(config["seed"]),
        num_layers=num_layers,
        latent_rank=int(config["latent_rank"]),
        group_size=int(config["group_size"]),
        alpha=result.alpha,
        prefix_tokens=int(config["prefix_tokens"]),
        recent_tokens=int(config["recent_tokens"]),
        clip_ratios=tuple(
            result.layers[layer].clip_ratio for layer in range(num_layers)
        ),
    )
    artifact = write_rotation_artifact(
        args.output_dir,
        metadata,
        {layer: layer_result.rotation for layer, layer_result in result.layers.items()},
    )
    print(
        json.dumps(
            {
                "alpha": result.alpha,
                "normalized_loss": result.normalized_loss,
                "manifest_sha256": artifact.manifest_sha256,
                "rotations_sha256": artifact.rotations_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
