import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.calibration import (
    bit_reversal_permutation,
    normalized_hadamard,
    oscar_covariance_rotation,
)
from vllm.model_executor.layers.quantization.oscar_mla.fit import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_CLIP_GRID,
    LayerCaptureStatistics,
    merge_capture_payloads,
    search_shared_rotations,
)


def _payload(tp_rank: int, *, layer_name: str = "model.layers.3.attn") -> dict:
    rank = 8
    generator = torch.Generator().manual_seed(101 + tp_rank)
    score_rows = torch.randn(12, rank, generator=generator, dtype=torch.float64)
    value_rows = torch.randn(12, rank, generator=generator, dtype=torch.float64)
    latent_rows = torch.randn(12, rank, generator=generator, dtype=torch.float64)
    payload = {
        "format_version": 1,
        "layer_name": layer_name,
        "split": "holdout",
        "tp_rank": tp_rank,
        "latent_rank": rank,
        "token_budget": 12,
        "captured_tokens": 12,
        "score_covariance_samples": 12,
        "value_covariance_samples": 12,
        "score_second_moment_sum": score_rows.T @ score_rows,
        "value_second_moment_sum": value_rows.T @ value_rows,
        "latent_samples": latent_rows[:4].float(),
    }
    if tp_rank == 0:
        payload["latent_covariance_samples"] = 12
        payload["latent_second_moment_sum"] = latent_rows.T @ latent_rows
    return payload


def test_oscar_rotation_components_are_orthogonal_and_deterministic() -> None:
    generator = torch.Generator().manual_seed(103)
    rows = torch.randn(64, 8, generator=generator, dtype=torch.float64)
    covariance = rows.T @ rows

    hadamard = normalized_hadamard(8)
    permutation = bit_reversal_permutation(8)
    rotation = oscar_covariance_rotation(covariance)

    torch.testing.assert_close(hadamard.T @ hadamard, torch.eye(8, dtype=torch.float64))
    assert permutation.tolist() == [0, 4, 2, 6, 1, 5, 3, 7]
    torch.testing.assert_close(
        rotation.T @ rotation,
        torch.eye(8, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(rotation, oscar_covariance_rotation(covariance))


def test_merge_capture_payloads_uses_all_tp_covariance_and_rank_zero_latent() -> None:
    payloads = [_payload(0), _payload(1)]

    merged = merge_capture_payloads(
        payloads,
        layer_name="model.layers.3.attn",
        tp_size=2,
        split="holdout",
        latent_rank=8,
        token_budget=12,
    )

    expected_score = (
        sum(payload["score_second_moment_sum"] for payload in payloads) / 24
    )
    torch.testing.assert_close(merged.score_covariance, expected_score)
    torch.testing.assert_close(merged.latent_samples, payloads[0]["latent_samples"])
    assert merged.score_samples == 24
    assert merged.value_samples == 24
    assert merged.latent_samples_count == 12


def test_merge_capture_payloads_rejects_missing_rank_and_wrong_identity() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        merge_capture_payloads(
            [_payload(0)],
            layer_name="model.layers.3.attn",
            tp_size=2,
            split="holdout",
            latent_rank=8,
            token_budget=12,
        )

    with pytest.raises(ValueError, match="layer_name mismatch"):
        merge_capture_payloads(
            [_payload(0, layer_name="wrong"), _payload(1, layer_name="wrong")],
            layer_name="model.layers.3.attn",
            tp_size=2,
            split="holdout",
            latent_rank=8,
            token_budget=12,
        )


def _statistics(seed: int) -> LayerCaptureStatistics:
    generator = torch.Generator().manual_seed(seed)
    score_rows = torch.randn(256, 128, generator=generator, dtype=torch.float64)
    value_rows = torch.randn(256, 128, generator=generator, dtype=torch.float64)
    latent = torch.randn(64, 128, generator=generator)
    return LayerCaptureStatistics(
        score_covariance=score_rows.T @ score_rows / score_rows.shape[0],
        value_covariance=value_rows.T @ value_rows / value_rows.shape[0],
        latent_covariance=latent.double().T @ latent.double() / latent.shape[0],
        latent_samples=latent,
        score_samples=256,
        value_samples=256,
        latent_samples_count=64,
    )


def test_shared_search_uses_only_fixed_grids_and_returns_orthogonal_layers() -> None:
    train = {0: _statistics(107), 1: _statistics(109)}
    holdout = {0: _statistics(113), 1: _statistics(127)}

    result = search_shared_rotations(train, holdout, group_size=128)

    assert result.alpha in DEFAULT_ALPHA_GRID
    assert set(result.alpha_losses) == set(DEFAULT_ALPHA_GRID)
    assert result.normalized_loss == min(result.alpha_losses.values())
    assert set(result.layers) == {0, 1}
    for layer in result.layers.values():
        assert layer.clip_ratio in DEFAULT_CLIP_GRID
        torch.testing.assert_close(
            layer.rotation.T @ layer.rotation,
            torch.eye(128),
            atol=1e-5,
            rtol=1e-5,
        )


def test_shared_search_rejects_layer_mismatch_and_empty_holdout() -> None:
    statistics = _statistics(131)
    with pytest.raises(ValueError, match="non-empty and equal"):
        search_shared_rotations({0: statistics}, {1: statistics}, group_size=128)

    empty = LayerCaptureStatistics(
        **{**statistics.__dict__, "latent_samples": torch.empty(0, 128)}
    )
    with pytest.raises(ValueError, match="must not be empty"):
        search_shared_rotations({0: statistics}, {0: empty}, group_size=128)
