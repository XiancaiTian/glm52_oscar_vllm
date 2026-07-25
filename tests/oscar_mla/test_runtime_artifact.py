import json

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.artifact import (
    ArtifactMetadata,
    write_rotation_artifact,
)
from vllm.model_executor.layers.quantization.oscar_mla.runtime import (
    _ARTIFACT_ENV,
    _EXPECTATION_ENV,
    _load_runtime_artifact,
    load_layer_runtime_parameters,
)


def _metadata() -> ArtifactMetadata:
    return ArtifactMetadata(
        model_config_sha256="1" * 64,
        checkpoint_manifest_sha256="2" * 64,
        expert_mapping_sha256="3" * 64,
        calibration_code_commit="4" * 40,
        calibration_manifest_sha256="5" * 64,
        seed=7,
        num_layers=2,
        latent_rank=4,
        group_size=2,
        alpha=0.5,
        prefix_tokens=2,
        recent_tokens=3,
        clip_ratios=(0.92, 0.98),
    )


def _expectation(metadata: ArtifactMetadata) -> dict[str, object]:
    return {
        "model_config_sha256": metadata.model_config_sha256,
        "checkpoint_manifest_sha256": metadata.checkpoint_manifest_sha256,
        "expert_mapping_sha256": metadata.expert_mapping_sha256,
        "num_layers": metadata.num_layers,
        "latent_rank": metadata.latent_rank,
        "group_size": metadata.group_size,
        "prefix_tokens": metadata.prefix_tokens,
        "recent_tokens": metadata.recent_tokens,
    }


def test_runtime_artifact_binds_layer_and_identity(tmp_path, monkeypatch) -> None:
    metadata = _metadata()
    artifact_dir = tmp_path / "artifact"
    expected_rotations = {
        0: torch.eye(4),
        1: torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    }
    written = write_rotation_artifact(artifact_dir, metadata, expected_rotations)
    expectation_path = tmp_path / "expectation.json"
    expectation_path.write_text(
        json.dumps(_expectation(metadata)),
        encoding="utf-8",
    )
    monkeypatch.setenv(_ARTIFACT_ENV, str(artifact_dir))
    monkeypatch.setenv(_EXPECTATION_ENV, str(expectation_path))
    _load_runtime_artifact.cache_clear()

    result = load_layer_runtime_parameters(
        "model.layers.1.self_attn.mla_attn",
        latent_rank=4,
        prefix_tokens=2,
        recent_tokens=3,
    )

    torch.testing.assert_close(result.rotation, expected_rotations[1])
    assert result.clip_ratio == 0.98
    assert result.manifest_sha256 == written.manifest_sha256
    assert result.rotations_sha256 == written.rotations_sha256


def test_runtime_artifact_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(_ARTIFACT_ENV, raising=False)
    monkeypatch.delenv(_EXPECTATION_ENV, raising=False)
    with pytest.raises(ValueError, match="requires"):
        load_layer_runtime_parameters(
            "model.layers.0.self_attn",
            latent_rank=4,
            prefix_tokens=2,
            recent_tokens=3,
        )

    metadata = _metadata()
    artifact_dir = tmp_path / "artifact"
    write_rotation_artifact(
        artifact_dir,
        metadata,
        {0: torch.eye(4), 1: torch.eye(4)},
    )
    expectation = _expectation(metadata)
    expectation["model_config_sha256"] = "a" * 64
    expectation_path = tmp_path / "expectation.json"
    expectation_path.write_text(json.dumps(expectation), encoding="utf-8")
    monkeypatch.setenv(_ARTIFACT_ENV, str(artifact_dir))
    monkeypatch.setenv(_EXPECTATION_ENV, str(expectation_path))
    _load_runtime_artifact.cache_clear()

    with pytest.raises(ValueError, match="does not match runtime"):
        load_layer_runtime_parameters(
            "model.layers.0.self_attn",
            latent_rank=4,
            prefix_tokens=2,
            recent_tokens=3,
        )
