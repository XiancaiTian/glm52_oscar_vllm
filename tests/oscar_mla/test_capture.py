import json

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.capture import (
    ActivationCaptureSession,
    CaptureConfig,
)


def test_capture_config_loads_and_validates(tmp_path) -> None:
    config_path = tmp_path / "capture.json"
    config_path.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path / "output"),
                "token_budget": 16,
                "latent_rank": 8,
                "seed": 71,
                "split": "train",
                "reservoir_rows": 4,
                "dsa_sample_rows": 2,
            }
        ),
        encoding="utf-8",
    )

    config = CaptureConfig.from_json(config_path)

    assert config.token_budget == 16
    assert config.reservoir_rows == 4
    with pytest.raises(ValueError, match="dsa_sample_rows"):
        CaptureConfig(
            output_dir=str(tmp_path),
            token_budget=4,
            latent_rank=8,
            seed=71,
            split="holdout",
            reservoir_rows=1,
            dsa_sample_rows=2,
        )


def test_capture_is_read_only_and_writes_exact_statistics(tmp_path) -> None:
    config = CaptureConfig(
        output_dir=str(tmp_path),
        token_budget=6,
        latent_rank=4,
        seed=73,
        split="holdout",
        flush_interval_rows=3,
        reservoir_rows=3,
        dsa_sample_rows=2,
        tp_rank=2,
        capture_latent=True,
    )
    session = ActivationCaptureSession(config)
    generator = torch.Generator().manual_seed(79)
    latent = torch.randn(7, 4, generator=generator)
    query = torch.randn(7, 3, 4, generator=generator)
    value = torch.randn(7, 3, 4, generator=generator)
    dsa = torch.arange(35, dtype=torch.int32).reshape(7, 5)
    originals = tuple(tensor.clone() for tensor in (latent, query, value, dsa))
    pointers = tuple(tensor.data_ptr() for tensor in (latent, query, value, dsa))

    session.capture("model.layers.4.attn", latent[:4], query[:4], value[:4], dsa[:4])
    session.capture("model.layers.4.attn", latent[4:], query[4:], value[4:], dsa[4:])

    for tensor, original, pointer in zip(
        (latent, query, value, dsa),
        originals,
        pointers,
    ):
        torch.testing.assert_close(tensor, original)
        assert tensor.data_ptr() == pointer

    output_path = tmp_path / "tp_rank_02" / "layers" / "model.layers.4.attn.pt"
    payload = torch.load(output_path, weights_only=True)
    positions = torch.arange(6)
    heads = positions.remainder(query.shape[1])
    expected_query = query[:6][positions, heads].double()
    expected_value = value[:6][positions, heads].double()
    expected_latent = latent[:6].double()

    assert payload["captured_tokens"] == 6
    assert payload["score_covariance_samples"] == 6
    assert payload["value_covariance_samples"] == 6
    assert payload["latent_covariance_samples"] == 6
    torch.testing.assert_close(
        payload["score_second_moment_sum"],
        expected_query.T @ expected_query,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        payload["value_second_moment_sum"],
        expected_value.T @ expected_value,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        payload["latent_second_moment_sum"],
        expected_latent.T @ expected_latent,
        rtol=1e-5,
        atol=1e-5,
    )
    sample_positions = payload["sample_token_positions"]
    torch.testing.assert_close(
        payload["latent_samples"],
        latent[sample_positions],
    )
    torch.testing.assert_close(
        payload["query_samples"],
        query[sample_positions, sample_positions.remainder(query.shape[1])],
    )
    assert payload["dsa_samples"].shape == (2, 5)


def test_capture_rejects_wrong_latent_shape_without_writing(tmp_path) -> None:
    config = CaptureConfig(
        output_dir=str(tmp_path),
        token_budget=2,
        latent_rank=4,
        seed=83,
        split="train",
    )
    session = ActivationCaptureSession(config)

    with pytest.raises(ValueError, match="latent"):
        session.capture(
            "model.layers.0.attn",
            torch.zeros(2, 3),
            torch.zeros(2, 2, 4),
            torch.zeros(2, 2, 4),
        )

    assert not list(tmp_path.rglob("*.pt"))
