import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.reference import (
    dequantize_int2,
    mixed_latent_attention,
    native_latent_attention,
    pack_int2,
    partition_mixed_tokens,
    quantize_int2,
    rotated_latent_attention,
    unpack_int2,
)


def _orthogonal(dim: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    matrix = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(matrix).Q


def test_unquantized_rotation_matches_native_attention() -> None:
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(3, 2, 8, generator=generator, dtype=torch.float64)
    latent = torch.randn(7, 8, generator=generator, dtype=torch.float64)
    rotation = _orthogonal(8)

    expected = native_latent_attention(query, latent)
    actual = rotated_latent_attention(query, latent, rotation)

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_int2_pack_unpack_roundtrip_and_error_bound() -> None:
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(4, 16, generator=generator, dtype=torch.float32)

    quantized = quantize_int2(values, group_size=8, clip_ratio=0.99)
    packed = pack_int2(quantized.data)
    unpacked = unpack_int2(packed, original_dim=16)
    restored = dequantize_int2(
        unpacked,
        quantized.scale,
        quantized.zero_point,
        group_size=8,
        dtype=values.dtype,
    )

    assert packed.dtype == torch.uint8
    assert packed.shape == (4, 4)
    torch.testing.assert_close(unpacked, quantized.data)
    error = (restored - quantized.clipped).abs().reshape(4, 2, 8)
    assert torch.all(error <= quantized.scale / 2 + 1e-6)


def test_int2_constant_input_is_reconstructed() -> None:
    values = torch.full((2, 16), 3.25, dtype=torch.float32)
    quantized = quantize_int2(values, group_size=8, clip_ratio=0.99)
    restored = dequantize_int2(
        quantized.data,
        quantized.scale,
        quantized.zero_point,
        group_size=8,
        dtype=values.dtype,
    )

    torch.testing.assert_close(restored, values, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("seq_len", "expected"),
    [
        (63, (63, 0, 0)),
        (64, (64, 0, 0)),
        (319, (64, 255, 0)),
        (320, (64, 256, 0)),
        (321, (64, 256, 1)),
    ],
)
def test_mixed_token_boundaries(
    seq_len: int,
    expected: tuple[int, int, int],
) -> None:
    partition = partition_mixed_tokens(seq_len, prefix_tokens=64, recent_tokens=256)

    assert (
        partition.prefix.stop - partition.prefix.start,
        partition.recent.stop - partition.recent.start,
        partition.history.stop - partition.history.start,
    ) == expected
    assert partition.total_tokens == seq_len


def test_unquantized_mixed_attention_matches_native() -> None:
    generator = torch.Generator().manual_seed(47)
    query = torch.randn(2, 3, 8, generator=generator, dtype=torch.float64)
    latent = torch.randn(321, 8, generator=generator, dtype=torch.float64)
    rotation = _orthogonal(8)

    expected = native_latent_attention(query, latent)
    actual = mixed_latent_attention(
        query,
        prefix_latent=latent[:64],
        recent_latent=latent[65:],
        history_rotated=latent[64:65] @ rotation,
        rotation=rotation,
    )

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
