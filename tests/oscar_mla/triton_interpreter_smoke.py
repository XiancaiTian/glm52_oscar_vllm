"""CPU-interpreter smoke for OSCAR MLA Triton kernels.

This module runs in a fresh process with ``TRITON_INTERPRET=1``. It validates
Triton semantics without creating a CUDA context; SM80 compilation and A800
launch remain separate acceptance gates.
"""

import torch

from vllm.model_executor.layers.quantization.oscar_mla.reference import (
    mixed_latent_attention,
)
from vllm.v1.attention.ops import triton_oscar_mla_decode as decode
from vllm.v1.attention.ops import triton_oscar_mla_store as store

store._require_cuda_tensor = lambda *args, **kwargs: None
decode._require_cuda_tensor = lambda *args, **kwargs: None

torch.manual_seed(7)
latent_rank = 512
sequence_length = 5
latent = torch.randn(sequence_length, latent_rank, dtype=torch.bfloat16)
query = torch.randn(1, 1, latent_rank, dtype=torch.bfloat16)
rotation = torch.eye(latent_rank, dtype=torch.bfloat16)
prefix = torch.zeros(1, 2, latent_rank, dtype=torch.bfloat16)
recent = torch.zeros(1, 2, latent_rank, dtype=torch.bfloat16)
positions = torch.arange(sequence_length, dtype=torch.int32)
zero_index = torch.zeros(1, dtype=torch.int32)

store.oscar_mla_store_bf16(
    latent,
    prefix,
    recent,
    positions,
    torch.full_like(positions, sequence_length),
    torch.zeros_like(positions),
)
torch.testing.assert_close(prefix[0], latent[:2])
torch.testing.assert_close(recent[0, 0], latent[4])
torch.testing.assert_close(recent[0, 1], latent[3])

history_data = torch.zeros(
    1,
    16,
    latent_rank // 4,
    dtype=torch.uint8,
)
history_scale = torch.zeros(
    1,
    16,
    latent_rank // 128,
    dtype=torch.float32,
)
history_zero = torch.zeros_like(history_scale)
store.oscar_mla_rotate_quantize_store(
    latent[2:3],
    rotation,
    history_data,
    history_scale,
    history_zero,
    zero_index,
    zero_index,
    clip_ratio=0.96,
)
history = store.oscar_mla_dequantize_history(
    history_data,
    history_scale,
    history_zero,
    zero_index,
    zero_index,
)
output, lse = decode.oscar_mla_sparse_decode(
    query,
    torch.arange(sequence_length, dtype=torch.int32).unsqueeze(0),
    prefix,
    recent,
    history_data,
    history_scale,
    history_zero,
    torch.zeros(1, 1, dtype=torch.int32),
    zero_index,
    torch.tensor([sequence_length], dtype=torch.int32),
    rotation,
    num_splits=2,
)
expected = mixed_latent_attention(
    query.float(),
    prefix_latent=latent[:2].float(),
    recent_latent=latent[3:].float(),
    history_rotated=history,
    rotation=rotation.float(),
)
torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
assert bool(output.isfinite().all())
assert bool(lse.isfinite().all())

prefill_output, prefill_lse = decode.oscar_mla_sparse_prefill(
    query.repeat(2, 1, 1),
    torch.arange(sequence_length, dtype=torch.int32).repeat(2, 1),
    torch.zeros(2, dtype=torch.int32),
    torch.tensor([2, 4], dtype=torch.int32),
    prefix,
    recent,
    history_data,
    history_scale,
    history_zero,
    torch.zeros(1, 1, dtype=torch.int32),
    zero_index,
    torch.tensor([sequence_length], dtype=torch.int32),
    rotation,
    num_splits=2,
)
prefill_expected = torch.cat(
    (
        mixed_latent_attention(
            query.float(),
            prefix_latent=latent[:2].float(),
            recent_latent=latent[:0].float(),
            history_rotated=history,
            rotation=rotation.float(),
        ),
        expected,
    ),
    dim=0,
)
torch.testing.assert_close(
    prefill_output,
    prefill_expected,
    atol=1e-5,
    rtol=1e-5,
)
assert bool(prefill_output.isfinite().all())
assert bool(prefill_lse.isfinite().all())
print(
    "interpreter_smoke",
    f"latent_rank={latent_rank}",
    f"groups={history_scale.shape[-1]}",
    f"max_error={(output - expected).abs().max().item()}",
    f"prefill_max_error={(prefill_output - prefill_expected).abs().max().item()}",
)
