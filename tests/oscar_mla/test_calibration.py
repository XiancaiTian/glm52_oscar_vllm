import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.calibration import (
    CovarianceAccumulator,
    build_shared_covariance,
    covariance_rotation,
    normalize_covariance,
)


def test_covariance_accumulator_matches_direct_computation() -> None:
    generator = torch.Generator().manual_seed(53)
    first = torch.randn(5, 3, 8, generator=generator, dtype=torch.float32)
    second = torch.randn(7, 3, 8, generator=generator, dtype=torch.float32)
    accumulator = CovarianceAccumulator(8)

    accumulator.update(first)
    accumulator.update(second)

    flat = torch.cat([first.reshape(-1, 8), second.reshape(-1, 8)]).double()
    expected = flat.T @ flat / flat.shape[0]
    torch.testing.assert_close(accumulator.covariance(), expected)
    assert accumulator.samples == flat.shape[0]


def test_covariance_normalization_and_shared_weight() -> None:
    score = torch.diag(torch.arange(1, 9, dtype=torch.float64))
    value = torch.diag(torch.arange(8, 0, -1, dtype=torch.float64))

    score_normalized = normalize_covariance(score)
    value_normalized = normalize_covariance(value)
    shared = build_shared_covariance(score, value, alpha=0.25)

    torch.testing.assert_close(
        score_normalized.trace(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        value_normalized.trace(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        shared,
        0.25 * score_normalized + 0.75 * value_normalized,
    )


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_shared_covariance_rejects_invalid_alpha(alpha: float) -> None:
    covariance = torch.eye(8, dtype=torch.float64)

    with pytest.raises(ValueError, match="alpha"):
        build_shared_covariance(covariance, covariance, alpha=alpha)


def test_covariance_rotation_is_orthogonal() -> None:
    generator = torch.Generator().manual_seed(59)
    samples = torch.randn(128, 8, generator=generator, dtype=torch.float64)
    covariance = samples.T @ samples / samples.shape[0]

    rotation = covariance_rotation(covariance)

    torch.testing.assert_close(
        rotation.T @ rotation,
        torch.eye(8, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
