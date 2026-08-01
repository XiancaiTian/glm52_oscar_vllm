# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.reference import (
    mixed_latent_attention_with_lse,
)
from vllm.v1.attention.ops.triton_oscar_mla_decode import (
    _mixed_sparse_prefill_stage1,
    _prefill_head_block_size,
    oscar_mla_sparse_decode,
    oscar_mla_sparse_prefill,
)
from vllm.v1.attention.ops.triton_oscar_mla_store import (
    oscar_mla_dequantize_history,
    oscar_mla_rotate_quantize_store,
    oscar_mla_store_bf16,
)

RUN_CUDA_TESTS = os.environ.get("VLLM_OSCAR_RUN_CUDA_TESTS") == "1"
requires_cuda = pytest.mark.skipif(
    not RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="set VLLM_OSCAR_RUN_CUDA_TESTS=1 on an authorized idle GPU",
)


@pytest.mark.parametrize(
    ("num_heads", "expected"),
    [(1, 8), (8, 8), (16, 16), (17, 32), (32, 32)],
)
def test_prefill_head_block_size(num_heads: int, expected: int) -> None:
    assert _prefill_head_block_size(num_heads) == expected


def test_grouped_prefill_uses_causal_runtime_loop_bound() -> None:
    source = inspect.getsource(_mixed_sparse_prefill_stage1.fn)
    assert "effective_topk = tl.minimum(topk, causal_seq_len)" in source
    assert "tl.range(0, effective_topk, block_t)" in source


def test_grouped_prefill_skips_zero_bf16_tile_dots() -> None:
    source = inspect.getsource(_mixed_sparse_prefill_stage1.fn)
    assert "has_bf16 = tl.sum(is_bf16.to(tl.int32), axis=0) > 0" in source
    assert source.count("if has_bf16:") == 2


def test_triton_interpreter_smoke() -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(Path(__file__).parents[2]),
            "TRANSFORMERS_OFFLINE": "1",
            "TRITON_INTERPRET": "1",
        }
    )
    script = Path(__file__).with_name("triton_interpreter_smoke.py")
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "latent_rank=512" in completed.stdout
    assert "max_error=" in completed.stdout
    assert "lse_max_error=" in completed.stdout
    assert "prefill_max_error=" in completed.stdout
    assert "prefill_lse_max_error=" in completed.stdout
    assert "multi_request_max_error=" in completed.stdout
    assert "multi_request_lse_max_error=" in completed.stdout


def _rotation(dim: int, *, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(41)
    matrix = torch.randn(dim, dim, generator=generator, device=device)
    q, _ = torch.linalg.qr(matrix.float())
    return q.to(torch.bfloat16)


def _assert_oracle(
    output: torch.Tensor,
    lse: torch.Tensor,
    expected: torch.Tensor,
    expected_lse: torch.Tensor,
    *,
    label: str,
) -> None:
    output_error = (output - expected).abs()
    lse_error = (lse - expected_lse).abs()
    print(
        f"{label} "
        f"output_max_error={output_error.max().item()} "
        f"output_mean_error={output_error.mean().item()} "
        f"lse_max_error={lse_error.max().item()} "
        f"lse_mean_error={lse_error.mean().item()}"
    )
    torch.testing.assert_close(output, expected, atol=0.5, rtol=0.03)
    torch.testing.assert_close(lse, expected_lse, atol=0.05, rtol=0.01)


def _pack_rope_cache(
    values: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = (values.shape[0] + block_size - 1) // block_size
    cache = torch.zeros(
        num_blocks,
        block_size,
        values.shape[1],
        dtype=torch.bfloat16,
        device=values.device,
    )
    cache.view(-1, values.shape[1])[: values.shape[0]].copy_(values)
    block_table = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device=values.device,
    ).unsqueeze(0)
    return cache, block_table


@requires_cuda
@pytest.mark.parametrize("seq_len", [63, 64, 319, 320, 321])
@pytest.mark.parametrize("num_heads", [1, 4])
def test_sparse_decode_matches_three_pool_oracle(
    seq_len: int,
    num_heads: int,
) -> None:
    device = torch.device("cuda")
    dim = 512
    group_size = 128
    prefix_tokens = 64
    recent_tokens = 256
    block_size = 16
    generator = torch.Generator(device=device).manual_seed(53 + seq_len + num_heads)
    latent = torch.randn(
        seq_len,
        dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    query = torch.randn(
        1,
        num_heads,
        dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    rope_values = torch.randn(
        seq_len,
        64,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    query_rope = torch.randn(
        1,
        num_heads,
        64,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    rope_cache, rope_block_table = _pack_rope_cache(rope_values, block_size)
    rotation = _rotation(dim, device=device)
    prefix = torch.zeros(
        1,
        prefix_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    recent = torch.zeros(
        1,
        recent_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    final_lens = torch.full_like(positions, seq_len)
    hp_rows_for_tokens = torch.zeros_like(positions)
    oscar_mla_store_bf16(
        latent,
        prefix,
        recent,
        positions,
        final_lens,
        hp_rows_for_tokens,
    )

    history_start = prefix_tokens
    history_end = max(prefix_tokens, seq_len - recent_tokens)
    history_len = history_end - history_start
    history_pages = max(1, (history_len + block_size - 1) // block_size)
    data = torch.zeros(
        history_pages,
        block_size,
        dim // 4,
        dtype=torch.uint8,
        device=device,
    )
    scale = torch.zeros(
        history_pages,
        block_size,
        dim // group_size,
        dtype=torch.float32,
        device=device,
    )
    zero = torch.zeros_like(scale)
    if history_len:
        history_latent = latent[history_start:history_end]
        history_indices = torch.arange(
            history_len,
            dtype=torch.int32,
            device=device,
        )
        page_ids = history_indices // block_size
        page_offsets = history_indices % block_size
        oscar_mla_rotate_quantize_store(
            history_latent,
            rotation,
            data,
            scale,
            zero,
            page_ids,
            page_offsets,
            clip_ratio=0.96,
        )
        history_rotated = oscar_mla_dequantize_history(
            data,
            scale,
            zero,
            page_ids,
            page_offsets,
        )
    else:
        history_rotated = torch.empty(
            0,
            dim,
            dtype=torch.float32,
            device=device,
        )

    selected = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)
    history_page_table = torch.arange(
        history_pages,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    hp_rows = torch.zeros(1, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    output, lse = oscar_mla_sparse_decode(
        query,
        query_rope,
        selected,
        prefix,
        recent,
        rope_cache,
        rope_block_table,
        data,
        scale,
        zero,
        history_page_table,
        hp_rows,
        seq_lens,
        rotation,
        num_splits=4,
    )

    prefix_end = min(seq_len, prefix_tokens)
    recent_start = max(prefix_end, seq_len - recent_tokens)
    expected, expected_lse = mixed_latent_attention_with_lse(
        query.float(),
        prefix_latent=latent[:prefix_end].float(),
        recent_latent=latent[recent_start:].float(),
        history_rotated=history_rotated,
        rotation=rotation.float(),
        query_rope=query_rope.float(),
        prefix_rope=rope_values[:prefix_end].float(),
        history_rope=rope_values[prefix_end:recent_start].float(),
        recent_rope=rope_values[recent_start:].float(),
    )
    _assert_oracle(
        output,
        lse,
        expected,
        expected_lse,
        label=f"decode_heads={num_heads}_seq={seq_len}",
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()


@requires_cuda
def test_sparse_decode_respects_selected_token_ids() -> None:
    device = torch.device("cuda")
    dim = 512
    seq_len = 321
    generator = torch.Generator(device=device).manual_seed(71)
    latent = torch.randn(
        seq_len,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        1,
        1,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_values = torch.randn(
        seq_len,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_rope = torch.randn(
        1,
        1,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_cache, rope_block_table = _pack_rope_cache(rope_values, 16)
    rotation = _rotation(dim, device=device)
    prefix = torch.zeros(1, 64, dim, dtype=torch.bfloat16, device=device)
    recent = torch.zeros(1, 256, dim, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    oscar_mla_store_bf16(
        latent,
        prefix,
        recent,
        positions,
        torch.full_like(positions, seq_len),
        torch.zeros_like(positions),
    )
    data = torch.zeros(1, 16, dim // 4, dtype=torch.uint8, device=device)
    scale = torch.zeros(1, 16, 4, dtype=torch.float32, device=device)
    zero = torch.zeros_like(scale)
    oscar_mla_rotate_quantize_store(
        latent[64:65],
        rotation,
        data,
        scale,
        zero,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        clip_ratio=1.0,
    )
    history_rotated = oscar_mla_dequantize_history(
        data,
        scale,
        zero,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    selected = torch.tensor([[0, 64, 320]], dtype=torch.int32, device=device)

    output, lse = oscar_mla_sparse_decode(
        query,
        query_rope,
        selected,
        prefix,
        recent,
        rope_cache,
        rope_block_table,
        data,
        scale,
        zero,
        torch.zeros(1, 1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        rotation,
        num_splits=3,
    )

    expected, expected_lse = mixed_latent_attention_with_lse(
        query.float(),
        prefix_latent=latent[0:1].float(),
        recent_latent=latent[320:321].float(),
        history_rotated=history_rotated,
        rotation=rotation.float(),
        query_rope=query_rope.float(),
        prefix_rope=rope_values[0:1].float(),
        history_rope=rope_values[64:65].float(),
        recent_rope=rope_values[320:321].float(),
    )
    _assert_oracle(
        output,
        lse,
        expected,
        expected_lse,
        label="decode_selected_ids",
    )


@requires_cuda
@pytest.mark.parametrize("batch_size", [4, 8])
def test_sparse_decode_isolates_batched_requests(batch_size: int) -> None:
    device = torch.device("cuda")
    dim = 512
    seq_len = 321
    block_size = 16
    blocks_per_request = (seq_len + block_size - 1) // block_size
    generator = torch.Generator(device=device).manual_seed(83 + batch_size)
    latent = torch.randn(
        batch_size,
        seq_len,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        batch_size,
        1,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_values = torch.randn(
        batch_size,
        seq_len,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_rope = torch.randn(
        batch_size,
        1,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rotation = _rotation(dim, device=device)
    prefix = torch.zeros(
        batch_size,
        64,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    recent = torch.zeros(
        batch_size,
        256,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(
        batch_size
    )
    hp_rows_for_tokens = torch.arange(
        batch_size,
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(seq_len)
    oscar_mla_store_bf16(
        latent.flatten(0, 1),
        prefix,
        recent,
        positions,
        torch.full_like(positions, seq_len),
        hp_rows_for_tokens,
    )

    rope_cache = torch.zeros(
        batch_size * blocks_per_request,
        block_size,
        64,
        dtype=torch.bfloat16,
        device=device,
    )
    for request_index in range(batch_size):
        request_cache = rope_cache[
            request_index * blocks_per_request : (request_index + 1)
            * blocks_per_request
        ]
        request_cache.view(-1, 64)[:seq_len].copy_(rope_values[request_index])
    rope_block_table = torch.arange(
        batch_size * blocks_per_request,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, blocks_per_request)

    history_data = torch.zeros(
        batch_size,
        block_size,
        dim // 4,
        dtype=torch.uint8,
        device=device,
    )
    history_scale = torch.zeros(
        batch_size,
        block_size,
        dim // 128,
        dtype=torch.float32,
        device=device,
    )
    history_zero = torch.zeros_like(history_scale)
    page_ids = torch.arange(batch_size, dtype=torch.int32, device=device)
    page_offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)
    oscar_mla_rotate_quantize_store(
        latent[:, 64],
        rotation,
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
        clip_ratio=0.96,
    )
    history_rotated = oscar_mla_dequantize_history(
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
    )
    selected = torch.tensor(
        [0, 64, 320],
        dtype=torch.int32,
        device=device,
    ).repeat(batch_size, 1)
    output, lse = oscar_mla_sparse_decode(
        query,
        query_rope,
        selected,
        prefix,
        recent,
        rope_cache,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        page_ids.unsqueeze(1),
        torch.arange(batch_size, dtype=torch.int32, device=device),
        torch.full((batch_size,), seq_len, dtype=torch.int32, device=device),
        rotation,
        num_splits=3,
    )

    expected_rows = []
    expected_lse_rows = []
    for request_index in range(batch_size):
        expected_row, expected_lse_row = mixed_latent_attention_with_lse(
            query[request_index : request_index + 1].float(),
            prefix_latent=latent[request_index, 0:1].float(),
            recent_latent=latent[request_index, 320:321].float(),
            history_rotated=history_rotated[request_index : request_index + 1],
            rotation=rotation.float(),
            query_rope=query_rope[request_index : request_index + 1].float(),
            prefix_rope=rope_values[request_index, 0:1].float(),
            history_rope=rope_values[request_index, 64:65].float(),
            recent_rope=rope_values[request_index, 320:321].float(),
        )
        expected_rows.append(expected_row)
        expected_lse_rows.append(expected_lse_row)
    _assert_oracle(
        output,
        lse,
        torch.cat(expected_rows),
        torch.cat(expected_lse_rows),
        label=f"decode_batch={batch_size}",
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()


@requires_cuda
@pytest.mark.parametrize("num_queries", [1, 4, 8])
@pytest.mark.parametrize("num_heads", [1, 8])
def test_sparse_prefill_is_causal_and_matches_three_pool_oracle(
    num_queries: int,
    num_heads: int,
) -> None:
    device = torch.device("cuda")
    dim = 512
    seq_len = 321
    block_size = 16
    generator = torch.Generator(device=device).manual_seed(97 + num_queries)
    latent = torch.randn(
        seq_len,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        num_queries,
        num_heads,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_values = torch.randn(
        seq_len,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_rope = torch.randn(
        num_queries,
        num_heads,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_cache, rope_block_table = _pack_rope_cache(rope_values, block_size)
    rotation = _rotation(dim, device=device)
    prefix = torch.zeros(1, 64, dim, dtype=torch.bfloat16, device=device)
    recent = torch.zeros(1, 256, dim, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    oscar_mla_store_bf16(
        latent,
        prefix,
        recent,
        positions,
        torch.full_like(positions, seq_len),
        torch.zeros_like(positions),
    )

    history_data = torch.zeros(
        1,
        block_size,
        dim // 4,
        dtype=torch.uint8,
        device=device,
    )
    history_scale = torch.zeros(
        1,
        block_size,
        dim // 128,
        dtype=torch.float32,
        device=device,
    )
    history_zero = torch.zeros_like(history_scale)
    zero_index = torch.zeros(1, dtype=torch.int32, device=device)
    oscar_mla_rotate_quantize_store(
        latent[64:65],
        rotation,
        history_data,
        history_scale,
        history_zero,
        zero_index,
        zero_index,
        clip_ratio=0.96,
    )
    history_rotated = oscar_mla_dequantize_history(
        history_data,
        history_scale,
        history_zero,
        zero_index,
        zero_index,
    )

    if num_queries == 1:
        query_positions = torch.tensor([320], dtype=torch.int32, device=device)
    elif num_queries == 4:
        query_positions = torch.tensor(
            [63, 64, 319, 320],
            dtype=torch.int32,
            device=device,
        )
    else:
        query_positions = torch.tensor(
            [63, 64, 100, 150, 200, 250, 319, 320],
            dtype=torch.int32,
            device=device,
        )
    selected = positions.unsqueeze(0).expand(num_queries, -1)
    output, lse = oscar_mla_sparse_prefill(
        query,
        query_rope,
        selected,
        torch.zeros(num_queries, dtype=torch.int32, device=device),
        query_positions,
        prefix,
        recent,
        rope_cache,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        zero_index.view(1, 1),
        zero_index,
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        rotation,
        inverse_rotation=rotation.T.contiguous(),
        num_splits=1,
    )

    expected_rows = []
    expected_lse_rows = []
    for row, query_position in enumerate(query_positions.tolist()):
        causal_length = query_position + 1
        expected_row, expected_lse_row = mixed_latent_attention_with_lse(
            query[row : row + 1].float(),
            prefix_latent=latent[: min(64, causal_length)].float(),
            recent_latent=latent[65:causal_length].float(),
            history_rotated=(
                history_rotated if causal_length > 64 else history_rotated[:0]
            ),
            rotation=rotation.float(),
            query_rope=query_rope[row : row + 1].float(),
            prefix_rope=rope_values[: min(64, causal_length)].float(),
            history_rope=(
                rope_values[64:65] if causal_length > 64 else rope_values[:0]
            ).float(),
            recent_rope=rope_values[65:causal_length].float(),
        )
        expected_rows.append(expected_row)
        expected_lse_rows.append(expected_lse_row)
    expected = torch.cat(expected_rows, dim=0)
    expected_lse = torch.cat(expected_lse_rows, dim=0)
    _assert_oracle(
        output,
        lse,
        expected,
        expected_lse,
        label=f"prefill_batch={num_queries}_heads={num_heads}",
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()
