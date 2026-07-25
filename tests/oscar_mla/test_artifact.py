import json

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.artifact import (
    ArtifactExpectation,
    ArtifactMetadata,
    load_rotation_artifact,
    write_rotation_artifact,
)


def _metadata() -> ArtifactMetadata:
    return ArtifactMetadata(
        model_config_sha256="1" * 64,
        checkpoint_manifest_sha256="2" * 64,
        expert_mapping_sha256="3" * 64,
        calibration_code_commit="4" * 40,
        calibration_manifest_sha256="5" * 64,
        seed=89,
        num_layers=2,
        latent_rank=8,
        group_size=4,
        alpha=0.5,
        prefix_tokens=64,
        recent_tokens=256,
        clip_ratios=(0.96, 0.98),
    )


def _rotations() -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(97)
    return {
        layer: torch.linalg.qr(
            torch.randn(8, 8, generator=generator, dtype=torch.float64)
        ).Q
        for layer in range(2)
    }


def _expectation() -> ArtifactExpectation:
    metadata = _metadata()
    return ArtifactExpectation(
        model_config_sha256=metadata.model_config_sha256,
        checkpoint_manifest_sha256=metadata.checkpoint_manifest_sha256,
        expert_mapping_sha256=metadata.expert_mapping_sha256,
        num_layers=metadata.num_layers,
        latent_rank=metadata.latent_rank,
        group_size=metadata.group_size,
        prefix_tokens=metadata.prefix_tokens,
        recent_tokens=metadata.recent_tokens,
    )


def test_artifact_round_trip_and_hashes(tmp_path) -> None:
    written = write_rotation_artifact(tmp_path, _metadata(), _rotations())

    loaded = load_rotation_artifact(tmp_path, expectation=_expectation())

    assert loaded.metadata == _metadata()
    assert loaded.manifest_sha256 == written.manifest_sha256
    assert loaded.rotations_sha256 == written.rotations_sha256
    assert set(loaded.rotations) == {0, 1}
    for rotation in loaded.rotations.values():
        torch.testing.assert_close(
            rotation.T @ rotation,
            torch.eye(8),
            atol=1e-5,
            rtol=1e-5,
        )


def test_artifact_rejects_missing_wrong_shape_and_nonorthogonal_layers(
    tmp_path,
) -> None:
    rotations = _rotations()
    with pytest.raises(ValueError, match="missing"):
        write_rotation_artifact(tmp_path / "missing", _metadata(), {0: rotations[0]})

    rotations[1] = torch.eye(7)
    with pytest.raises(ValueError, match="shape"):
        write_rotation_artifact(tmp_path / "shape", _metadata(), rotations)

    rotations[1] = torch.ones(8, 8)
    with pytest.raises(ValueError, match="orthogonal"):
        write_rotation_artifact(tmp_path / "orthogonal", _metadata(), rotations)


def test_artifact_rejects_runtime_identity_mismatch(tmp_path) -> None:
    write_rotation_artifact(tmp_path, _metadata(), _rotations())
    expectation = _expectation()
    wrong = ArtifactExpectation(
        **{
            **expectation.__dict__,
            "model_config_sha256": "f" * 64,
            "latent_rank": 16,
        }
    )

    with pytest.raises(ValueError, match="model_config_sha256, latent_rank"):
        load_rotation_artifact(tmp_path, expectation=wrong)


def test_artifact_rejects_tampered_tensor_and_layer_manifest(tmp_path) -> None:
    write_rotation_artifact(tmp_path, _metadata(), _rotations())
    rotation_path = tmp_path / "rotations.pt"
    rotation_path.write_bytes(rotation_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA256"):
        load_rotation_artifact(tmp_path)

    write_rotation_artifact(tmp_path, _metadata(), _rotations())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layer_ids"] = [0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="layer_ids"):
        load_rotation_artifact(tmp_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format_version": 2}, "format version"),
        ({"clip_ratios": (0.96,)}, "one value per layer"),
        ({"group_size": 3}, "exactly divide"),
        ({"calibration_manifest_sha256": "not-a-hash"}, "lowercase SHA256"),
    ],
)
def test_artifact_metadata_rejects_invalid_contract(changes, message) -> None:
    values = {**_metadata().__dict__, **changes}

    with pytest.raises(ValueError, match=message):
        ArtifactMetadata(**values)
