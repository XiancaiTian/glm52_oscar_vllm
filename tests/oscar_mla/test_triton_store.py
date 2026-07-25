import os

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.reference import (
    dequantize_int2,
    pack_int2,
    quantize_int2,
)
from vllm.v1.attention.ops.triton_oscar_mla_store import (
    _clip_index,
    oscar_mla_demote_recent,
    oscar_mla_dequantize_history,
    oscar_mla_rotate_quantize_store,
    oscar_mla_store_bf16,
    oscar_mla_store_rope,
)

RUN_CUDA_TESTS = os.environ.get("VLLM_OSCAR_RUN_CUDA_TESTS") == "1"
requires_cuda = pytest.mark.skipif(
    not RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="set VLLM_OSCAR_RUN_CUDA_TESTS=1 on an authorized idle GPU",
)


def _rotation(dim: int, *, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(17)
    matrix = torch.randn(dim, dim, generator=generator, device=device)
    q, _ = torch.linalg.qr(matrix.float())
    return q.to(torch.bfloat16)


def _history(
    *,
    pages: int,
    block_size: int,
    dim: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(
            pages,
            block_size,
            dim // 4,
            dtype=torch.uint8,
            device=device,
        ),
        torch.zeros(
            pages,
            block_size,
            dim // group_size,
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros(
            pages,
            block_size,
            dim // group_size,
            dtype=torch.float32,
            device=device,
        ),
    )


def test_clip_index_matches_reference_boundaries() -> None:
    assert _clip_index(0.92, 128) == 117
    assert _clip_index(0.99, 128) == 126
    assert _clip_index(1.0, 128) == 127
    with pytest.raises(ValueError, match="clip_ratio"):
        _clip_index(0.0, 128)


@requires_cuda
@pytest.mark.parametrize("num_rows", [1, 4, 17])
@pytest.mark.parametrize("clip_ratio", [0.92, 1.0])
def test_rotate_pack_and_dequant_match_pytorch(
    num_rows: int,
    clip_ratio: float,
) -> None:
    device = torch.device("cuda")
    dim = 512
    group_size = 128
    generator = torch.Generator(device=device).manual_seed(29 + num_rows)
    latent = torch.randn(
        num_rows,
        dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    rotation = _rotation(dim, device=device)
    data, scale, zero = _history(
        pages=4,
        block_size=16,
        dim=dim,
        group_size=group_size,
        device=device,
    )
    page_ids = (torch.arange(num_rows, device=device, dtype=torch.int32) // 16) + 1
    page_offsets = torch.arange(num_rows, device=device, dtype=torch.int32) % 16

    rotated = oscar_mla_rotate_quantize_store(
        latent,
        rotation,
        data,
        scale,
        zero,
        page_ids,
        page_offsets,
        clip_ratio=clip_ratio,
    )
    restored = oscar_mla_dequantize_history(
        data,
        scale,
        zero,
        page_ids,
        page_offsets,
    )

    expected_rotated = latent.float() @ rotation.float()
    expected_quantized = quantize_int2(
        rotated,
        group_size=group_size,
        clip_ratio=clip_ratio,
    )
    expected_restored = dequantize_int2(
        expected_quantized.data,
        expected_quantized.scale,
        expected_quantized.zero_point,
        group_size=group_size,
        dtype=torch.float32,
    )
    expected_packed = pack_int2(expected_quantized.data)

    torch.testing.assert_close(rotated, expected_rotated, atol=0.35, rtol=0.02)
    torch.testing.assert_close(restored, expected_restored, atol=0.35, rtol=0.02)
    torch.testing.assert_close(
        data[page_ids.long(), page_offsets.long()],
        expected_packed,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        scale[page_ids.long(), page_offsets.long()],
        expected_quantized.scale.squeeze(-1),
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        zero[page_ids.long(), page_offsets.long()],
        expected_quantized.zero_point.squeeze(-1),
        atol=2e-3,
        rtol=2e-3,
    )


@requires_cuda
def test_bf16_store_uses_final_partition_and_ring_addresses() -> None:
    device = torch.device("cuda")
    dim = 512
    prefix_tokens = 64
    recent_tokens = 256
    positions = torch.tensor(
        [0, 63, 64, 65, 319, 320, 321],
        dtype=torch.int32,
        device=device,
    )
    latent = torch.arange(
        positions.numel() * dim,
        dtype=torch.float32,
        device=device,
    ).view(-1, dim)
    prefix = torch.full(
        (1, prefix_tokens, dim),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    recent = torch.full(
        (1, recent_tokens, dim),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    final_lens = torch.full_like(positions, 322)
    hp_rows = torch.zeros_like(positions)

    oscar_mla_store_bf16(
        latent,
        prefix,
        recent,
        positions,
        final_lens,
        hp_rows,
    )

    torch.testing.assert_close(prefix[0, 0], latent[0].bfloat16())
    torch.testing.assert_close(prefix[0, 63], latent[1].bfloat16())
    assert torch.isnan(recent[0, 0]).all()
    torch.testing.assert_close(recent[0, 1], latent[3].bfloat16())
    torch.testing.assert_close(recent[0, 255], latent[4].bfloat16())
    torch.testing.assert_close(recent[0, 0], latent[5].bfloat16())
    torch.testing.assert_close(recent[0, 1], latent[6].bfloat16())


@requires_cuda
def test_rope_store_uses_standard_slot_mapping() -> None:
    device = torch.device("cuda")
    values = torch.arange(
        4 * 64,
        dtype=torch.float32,
        device=device,
    ).view(4, 1, 64)
    cache = torch.full(
        (3, 16, 64),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    slots = torch.tensor([0, 17, -1, 35], dtype=torch.int32, device=device)

    oscar_mla_store_rope(values, cache, slots)

    torch.testing.assert_close(cache[0, 0], values[0, 0].bfloat16())
    torch.testing.assert_close(cache[1, 1], values[1, 0].bfloat16())
    torch.testing.assert_close(cache[2, 3], values[3, 0].bfloat16())
    assert torch.isnan(cache[0, 1]).all()


@requires_cuda
def test_recent_demotion_matches_direct_history_store() -> None:
    device = torch.device("cuda")
    dim = 512
    group_size = 128
    rotation = _rotation(dim, device=device)
    recent = torch.randn(
        2,
        256,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.tensor([320, 321, 576], dtype=torch.int32, device=device)
    hp_rows = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    page_ids = torch.tensor([1, 1, 3], dtype=torch.int32, device=device)
    page_offsets = torch.tensor([0, 1, 7], dtype=torch.int32, device=device)
    data, scale, zero = _history(
        pages=4,
        block_size=16,
        dim=dim,
        group_size=group_size,
        device=device,
    )

    rotated = oscar_mla_demote_recent(
        recent,
        rotation,
        data,
        scale,
        zero,
        positions,
        hp_rows,
        page_ids,
        page_offsets,
        prefix_tokens=64,
        clip_ratio=0.96,
    )

    recent_indices = (positions - 64) % 256
    selected = recent[hp_rows.long(), recent_indices.long()]
    expected_rotated = selected.float() @ rotation.float()
    expected_quantized = quantize_int2(
        rotated,
        group_size=group_size,
        clip_ratio=0.96,
    )
    expected_restored = dequantize_int2(
        expected_quantized.data,
        expected_quantized.scale,
        expected_quantized.zero_point,
        group_size=group_size,
        dtype=torch.float32,
    )
    restored = oscar_mla_dequantize_history(
        data,
        scale,
        zero,
        page_ids,
        page_offsets,
    )
    torch.testing.assert_close(rotated, expected_rotated, atol=0.35, rtol=0.02)
    torch.testing.assert_close(restored, expected_restored, atol=0.35, rtol=0.02)


@requires_cuda
def test_chunked_and_one_shot_final_partitions_match() -> None:
    device = torch.device("cuda")
    dim = 512
    group_size = 128
    prefix_tokens = 64
    recent_tokens = 256
    final_length = 337
    first_chunk_length = 320
    generator = torch.Generator(device=device).manual_seed(47)
    latent = torch.randn(
        final_length,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rotation = _rotation(dim, device=device).T
    assert not rotation.is_contiguous()
    positions = torch.arange(final_length, dtype=torch.int32, device=device)
    hp_rows = torch.zeros(final_length, dtype=torch.int32, device=device)
    history_positions = positions[prefix_tokens : final_length - recent_tokens]
    page_ids = (history_positions - prefix_tokens) // 16
    page_offsets = (history_positions - prefix_tokens) % 16

    direct_data, direct_scale, direct_zero = _history(
        pages=2,
        block_size=16,
        dim=dim,
        group_size=group_size,
        device=device,
    )
    direct_prefix = torch.zeros(
        1,
        prefix_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    direct_recent = torch.zeros(
        1,
        recent_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    oscar_mla_rotate_quantize_store(
        latent[history_positions.long()],
        rotation,
        direct_data,
        direct_scale,
        direct_zero,
        page_ids,
        page_offsets,
        clip_ratio=0.96,
    )
    oscar_mla_store_bf16(
        latent,
        direct_prefix,
        direct_recent,
        positions,
        torch.full_like(positions, final_length),
        hp_rows,
    )

    chunked_data, chunked_scale, chunked_zero = _history(
        pages=2,
        block_size=16,
        dim=dim,
        group_size=group_size,
        device=device,
    )
    chunked_prefix = torch.zeros_like(direct_prefix)
    chunked_recent = torch.zeros_like(direct_recent)
    oscar_mla_store_bf16(
        latent[:first_chunk_length],
        chunked_prefix,
        chunked_recent,
        positions[:first_chunk_length],
        torch.full_like(positions[:first_chunk_length], first_chunk_length),
        hp_rows[:first_chunk_length],
    )
    oscar_mla_demote_recent(
        chunked_recent,
        rotation,
        chunked_data,
        chunked_scale,
        chunked_zero,
        history_positions,
        torch.zeros_like(history_positions),
        page_ids,
        page_offsets,
        prefix_tokens=prefix_tokens,
        clip_ratio=0.96,
    )
    oscar_mla_store_bf16(
        latent[first_chunk_length:],
        chunked_prefix,
        chunked_recent,
        positions[first_chunk_length:],
        torch.full_like(positions[first_chunk_length:], final_length),
        hp_rows[first_chunk_length:],
    )

    torch.testing.assert_close(chunked_data, direct_data, atol=0, rtol=0)
    torch.testing.assert_close(chunked_scale, direct_scale, atol=0, rtol=0)
    torch.testing.assert_close(chunked_zero, direct_zero, atol=0, rtol=0)
    torch.testing.assert_close(chunked_prefix, direct_prefix, atol=0, rtol=0)
    torch.testing.assert_close(chunked_recent, direct_recent, atol=0, rtol=0)
