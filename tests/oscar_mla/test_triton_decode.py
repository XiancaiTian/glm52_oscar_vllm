import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.reference import (
    mixed_latent_attention,
)
from vllm.v1.attention.ops.triton_oscar_mla_decode import (
    oscar_mla_sparse_decode,
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


def test_triton_interpreter_smoke() -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
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


def _rotation(dim: int, *, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(41)
    matrix = torch.randn(dim, dim, generator=generator, device=device)
    q, _ = torch.linalg.qr(matrix.float())
    return q.to(torch.bfloat16)


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
        selected,
        prefix,
        recent,
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
    expected = mixed_latent_attention(
        query.float(),
        prefix_latent=latent[:prefix_end].float(),
        recent_latent=latent[recent_start:].float(),
        history_rotated=history_rotated,
        rotation=rotation.float(),
    )
    torch.testing.assert_close(output, expected, atol=0.5, rtol=0.03)
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

    output, _ = oscar_mla_sparse_decode(
        query,
        selected,
        prefix,
        recent,
        data,
        scale,
        zero,
        torch.zeros(1, 1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        rotation,
        num_splits=3,
    )

    expected = mixed_latent_attention(
        query.float(),
        prefix_latent=latent[0:1].float(),
        recent_latent=latent[320:321].float(),
        history_rotated=history_rotated,
        rotation=rotation.float(),
    )
    torch.testing.assert_close(output, expected, atol=0.5, rtol=0.03)
