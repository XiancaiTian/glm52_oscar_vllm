# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 Triton writes and demotion for OSCAR shared-latent MLA caches."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rotate_latent_kernel(
    latent_ptr,
    rotation_ptr,
    output_ptr,
    num_rows,
    latent_rank: tl.constexpr,
    stride_latent_row: tl.constexpr,
    stride_latent_dim: tl.constexpr,
    stride_rotation_row: tl.constexpr,
    stride_rotation_col: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Compute ``latent @ rotation`` with an FP32 accumulator."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(latent_rank, block_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = pid_n * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    latent_ptrs = (
        latent_ptr + rows[:, None] * stride_latent_row + ks[None, :] * stride_latent_dim
    )
    rotation_ptrs = (
        rotation_ptr
        + ks[:, None] * stride_rotation_row
        + cols[None, :] * stride_rotation_col
    )
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(0, latent_rank, block_k):
        k_mask = k_start + ks < latent_rank
        latent = tl.load(
            latent_ptrs,
            mask=(rows[:, None] < num_rows) & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        rotation = tl.load(
            rotation_ptrs,
            mask=k_mask[:, None] & (cols[None, :] < latent_rank),
            other=0.0,
        ).to(tl.bfloat16)
        accumulator = tl.dot(latent, rotation, accumulator)
        latent_ptrs += block_k * stride_latent_dim
        rotation_ptrs += block_k * stride_rotation_row

    output_ptrs = (
        output_ptr
        + rows[:, None] * stride_output_row
        + cols[None, :] * stride_output_dim
    )
    tl.store(
        output_ptrs,
        accumulator,
        mask=(rows[:, None] < num_rows) & (cols[None, :] < latent_rank),
    )


@triton.jit
def _quantize_store_history_kernel(
    rotated_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    page_ids_ptr,
    page_offsets_ptr,
    num_rows,
    stride_rotated_row: tl.constexpr,
    stride_rotated_dim: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    clip_index: tl.constexpr,
):
    """Clip, asymmetrically quantize and pack one latent group."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    if row >= num_rows:
        return
    page = tl.load(page_ids_ptr + row)
    if page < 0:
        return
    token_offset = tl.load(page_offsets_ptr + row)

    dims = tl.arange(0, group_size)
    values = tl.load(
        rotated_ptr
        + row * stride_rotated_row
        + (group * group_size + dims) * stride_rotated_dim
    ).to(tl.float32)
    if clip_index >= 0:
        sorted_abs = tl.sort(tl.abs(values))
        threshold = tl.sum(
            tl.where(dims == clip_index, sorted_abs, 0.0),
            axis=0,
        )
        values = tl.minimum(tl.maximum(values, -threshold), threshold)

    value_min = tl.min(values, axis=0)
    value_max = tl.max(values, axis=0)
    scale = tl.maximum(value_max - value_min, 1e-8) / 3.0
    zero = -value_min / scale
    quantized = tl.minimum(
        tl.maximum((values / scale + zero + 0.5).to(tl.int32), 0),
        3,
    )
    quantized = tl.reshape(quantized, (packed_group_bytes, 4))
    shifts = tl.arange(0, 4) * 2
    packed = tl.sum(
        (quantized & 0x3) << shifts[None, :],
        axis=1,
    ).to(tl.uint8)
    packed_offsets = tl.arange(0, packed_group_bytes)
    data_base = (
        page * stride_data_page
        + token_offset * stride_data_token
        + group * packed_group_bytes * stride_data_byte
    )
    tl.store(
        history_data_ptr + data_base + packed_offsets * stride_data_byte,
        packed,
    )
    scale_base = (
        page * stride_scale_page
        + token_offset * stride_scale_token
        + group * stride_scale_group
    )
    zero_base = (
        page * stride_zero_page
        + token_offset * stride_zero_token
        + group * stride_zero_group
    )
    tl.store(history_scale_ptr + scale_base, scale)
    tl.store(history_zero_ptr + zero_base, zero)


@triton.jit
def _store_bf16_latent_kernel(
    latent_ptr,
    prefix_ptr,
    recent_ptr,
    logical_positions_ptr,
    final_seq_lens_ptr,
    hp_rows_ptr,
    num_rows,
    stride_latent_row: tl.constexpr,
    stride_latent_dim: tl.constexpr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_dim: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_dim: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    latent_rank: tl.constexpr,
    block_d: tl.constexpr,
):
    """Store only the final BF16 prefix/recent partition of each token."""
    row = tl.program_id(0)
    if row >= num_rows:
        return
    hp_row = tl.load(hp_rows_ptr + row)
    if hp_row < 0:
        return
    position = tl.load(logical_positions_ptr + row)
    seq_len = tl.load(final_seq_lens_ptr + row)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
    is_prefix = position < prefix_tokens
    is_recent = position >= recent_start
    if not (is_prefix or is_recent):
        return

    dims = tl.arange(0, block_d)
    mask = dims < latent_rank
    values = tl.load(
        latent_ptr + row * stride_latent_row + dims * stride_latent_dim,
        mask=mask,
    ).to(tl.bfloat16)
    prefix_base = hp_row * stride_prefix_row + position * stride_prefix_token
    tl.store(
        prefix_ptr + prefix_base + dims * stride_prefix_dim,
        values,
        mask=mask & is_prefix,
    )
    recent_idx = (position - prefix_tokens) % recent_tokens
    recent_base = hp_row * stride_recent_row + recent_idx * stride_recent_token
    tl.store(
        recent_ptr + recent_base + dims * stride_recent_dim,
        values,
        mask=mask & is_recent & ~is_prefix,
    )


@triton.jit
def _gather_recent_latent_kernel(
    recent_ptr,
    output_ptr,
    logical_positions_ptr,
    hp_rows_ptr,
    num_rows,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_dim: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_dim: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    latent_rank: tl.constexpr,
    block_d: tl.constexpr,
):
    """Gather the BF16 rows selected for recent-to-history demotion."""
    row = tl.program_id(0)
    if row >= num_rows:
        return
    hp_row = tl.load(hp_rows_ptr + row)
    position = tl.load(logical_positions_ptr + row)
    recent_idx = (position - prefix_tokens) % recent_tokens
    dims = tl.arange(0, block_d)
    mask = (dims < latent_rank) & (hp_row >= 0) & (position >= prefix_tokens)
    values = tl.load(
        recent_ptr
        + hp_row * stride_recent_row
        + recent_idx * stride_recent_token
        + dims * stride_recent_dim,
        mask=mask,
        other=0.0,
    )
    tl.store(
        output_ptr + row * stride_output_row + dims * stride_output_dim,
        values,
        mask=dims < latent_rank,
    )


@triton.jit
def _dequantize_history_kernel(
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    output_ptr,
    page_ids_ptr,
    page_offsets_ptr,
    num_rows,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_dim: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
):
    """Dequantize one packed history group into an FP32 oracle buffer."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    if row >= num_rows:
        return
    page = tl.load(page_ids_ptr + row)
    if page < 0:
        return
    token_offset = tl.load(page_offsets_ptr + row)

    dims = tl.arange(0, group_size)
    byte_offsets = dims // 4
    shifts = (dims % 4) * 2
    data_base = (
        page * stride_data_page
        + token_offset * stride_data_token
        + group * packed_group_bytes * stride_data_byte
    )
    packed = tl.load(history_data_ptr + data_base + byte_offsets * stride_data_byte).to(
        tl.int32
    )
    quantized = ((packed >> shifts) & 0x3).to(tl.float32)
    scale = tl.load(
        history_scale_ptr
        + page * stride_scale_page
        + token_offset * stride_scale_token
        + group * stride_scale_group
    ).to(tl.float32)
    zero = tl.load(
        history_zero_ptr
        + page * stride_zero_page
        + token_offset * stride_zero_token
        + group * stride_zero_group
    ).to(tl.float32)
    restored = (quantized - zero) * scale
    tl.store(
        output_ptr
        + row * stride_output_row
        + (group * group_size + dims) * stride_output_dim,
        restored,
    )


def _require_cuda_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    ndim: int,
    dtype: torch.dtype | tuple[torch.dtype, ...],
) -> None:
    dtypes = (dtype,) if isinstance(dtype, torch.dtype) else dtype
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape={tuple(tensor.shape)}")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {tensor.dtype}")


def _validate_history_tensors(
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
) -> tuple[int, int, int, int]:
    _require_cuda_tensor(
        history_data,
        name="history_data",
        ndim=3,
        dtype=torch.uint8,
    )
    _require_cuda_tensor(
        history_scale,
        name="history_scale",
        ndim=3,
        dtype=torch.float32,
    )
    _require_cuda_tensor(
        history_zero,
        name="history_zero",
        ndim=3,
        dtype=torch.float32,
    )
    if history_scale.shape != history_zero.shape:
        raise ValueError("history scale/zero shapes must match")
    if history_data.shape[:2] != history_scale.shape[:2]:
        raise ValueError("history data/metadata page geometry must match")
    num_groups = history_scale.shape[2]
    if num_groups <= 0:
        raise ValueError("history cache must contain at least one group")
    packed_bytes = history_data.shape[2]
    if packed_bytes % num_groups:
        raise ValueError("packed latent bytes must divide evenly across groups")
    packed_group_bytes = packed_bytes // num_groups
    group_size = packed_group_bytes * 4
    latent_rank = group_size * num_groups
    return num_groups, group_size, packed_group_bytes, latent_rank


def _validate_indices(
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    num_rows: int,
) -> None:
    for name, tensor in (("page_ids", page_ids), ("page_offsets", page_offsets)):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=1,
            dtype=(torch.int32, torch.int64),
        )
        if tensor.shape[0] != num_rows:
            raise ValueError(f"{name} length must equal the number of rows")


def _clip_index(clip_ratio: float, group_size: int) -> int:
    if not 0 < clip_ratio <= 1:
        raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")
    return min(int(clip_ratio * group_size), group_size - 1)


def oscar_mla_rotate(
    latent: torch.Tensor,
    rotation: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate shared latent rows with an SM80-compatible Triton matmul."""
    _require_cuda_tensor(
        latent,
        name="latent",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    _require_cuda_tensor(
        rotation,
        name="rotation",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    num_rows, latent_rank = latent.shape
    if rotation.shape != (latent_rank, latent_rank):
        raise ValueError("rotation must be square and match the latent rank")
    if latent.device != rotation.device:
        raise ValueError("latent and rotation must be on the same CUDA device")
    if output is None:
        output = torch.empty(
            (num_rows, latent_rank),
            dtype=torch.float32,
            device=latent.device,
        )
    else:
        _require_cuda_tensor(
            output,
            name="output",
            ndim=2,
            dtype=torch.float32,
        )
        if output.shape != latent.shape or output.device != latent.device:
            raise ValueError("rotation output shape/device must match latent")
    if num_rows == 0:
        return output

    block_m = 16
    block_n = 64
    block_k = 32
    grid = (triton.cdiv(num_rows, block_m) * triton.cdiv(latent_rank, block_n),)
    _rotate_latent_kernel[grid](
        latent,
        rotation,
        output,
        num_rows,
        latent_rank=latent_rank,
        stride_latent_row=latent.stride(0),
        stride_latent_dim=latent.stride(1),
        stride_rotation_row=rotation.stride(0),
        stride_rotation_col=rotation.stride(1),
        stride_output_row=output.stride(0),
        stride_output_dim=output.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=2,
    )
    return output


def oscar_mla_quantize_store_history(
    rotated: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    clip_ratio: float,
) -> None:
    """Clip and store already-rotated shared latent rows as grouped INT2."""
    _require_cuda_tensor(
        rotated,
        name="rotated",
        ndim=2,
        dtype=torch.float32,
    )
    num_groups, group_size, packed_group_bytes, latent_rank = _validate_history_tensors(
        history_data, history_scale, history_zero
    )
    num_rows = rotated.shape[0]
    if rotated.shape[1] != latent_rank:
        raise ValueError("rotated latent rank does not match history cache geometry")
    _validate_indices(page_ids, page_offsets, num_rows=num_rows)
    if not (
        rotated.device
        == history_data.device
        == history_scale.device
        == history_zero.device
        == page_ids.device
        == page_offsets.device
    ):
        raise ValueError("all OSCAR MLA history tensors must share one CUDA device")
    if num_rows == 0:
        return

    _quantize_store_history_kernel[(num_rows, num_groups)](
        rotated,
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
        num_rows,
        stride_rotated_row=rotated.stride(0),
        stride_rotated_dim=rotated.stride(1),
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        num_groups=num_groups,
        group_size=group_size,
        packed_group_bytes=packed_group_bytes,
        clip_index=_clip_index(clip_ratio, group_size),
        num_warps=4,
        num_stages=1,
    )


def oscar_mla_rotate_quantize_store(
    latent: torch.Tensor,
    rotation: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    clip_ratio: float,
    rotated: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate, clip, INT2-pack and store shared latent history rows."""
    rotated = oscar_mla_rotate(latent, rotation, output=rotated)
    oscar_mla_quantize_store_history(
        rotated,
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
        clip_ratio=clip_ratio,
    )
    return rotated


def oscar_mla_store_bf16(
    latent: torch.Tensor,
    prefix: torch.Tensor,
    recent: torch.Tensor,
    logical_positions: torch.Tensor,
    final_seq_lens: torch.Tensor,
    hp_rows: torch.Tensor,
) -> None:
    """Write rows belonging to the final BF16 prefix/recent partition."""
    _require_cuda_tensor(
        latent,
        name="latent",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    for name, tensor in (("prefix", prefix), ("recent", recent)):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=3,
            dtype=torch.bfloat16,
        )
    num_rows, latent_rank = latent.shape
    if prefix.shape[0] != recent.shape[0]:
        raise ValueError("prefix/recent row capacities must match")
    if prefix.shape[2] != latent_rank or recent.shape[2] != latent_rank:
        raise ValueError("BF16 cache latent rank must match input")
    if prefix.shape[1] <= 0 or recent.shape[1] <= 0:
        raise ValueError("BF16 prefix/recent windows must be positive")
    for name, tensor in (
        ("logical_positions", logical_positions),
        ("final_seq_lens", final_seq_lens),
        ("hp_rows", hp_rows),
    ):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=1,
            dtype=(torch.int32, torch.int64),
        )
        if tensor.shape[0] != num_rows:
            raise ValueError(f"{name} length must equal the number of latent rows")
    if not (
        latent.device
        == prefix.device
        == recent.device
        == logical_positions.device
        == final_seq_lens.device
        == hp_rows.device
    ):
        raise ValueError("all OSCAR MLA BF16 tensors must share one CUDA device")
    if num_rows == 0:
        return

    block_d = triton.next_power_of_2(latent_rank)
    _store_bf16_latent_kernel[(num_rows,)](
        latent,
        prefix,
        recent,
        logical_positions,
        final_seq_lens,
        hp_rows,
        num_rows,
        stride_latent_row=latent.stride(0),
        stride_latent_dim=latent.stride(1),
        stride_prefix_row=prefix.stride(0),
        stride_prefix_token=prefix.stride(1),
        stride_prefix_dim=prefix.stride(2),
        stride_recent_row=recent.stride(0),
        stride_recent_token=recent.stride(1),
        stride_recent_dim=recent.stride(2),
        prefix_tokens=prefix.shape[1],
        recent_tokens=recent.shape[1],
        latent_rank=latent_rank,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )


def oscar_mla_demote_recent(
    recent: torch.Tensor,
    rotation: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    logical_positions: torch.Tensor,
    hp_rows: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    prefix_tokens: int,
    clip_ratio: float,
) -> torch.Tensor:
    """Gather recent rows, rotate them, and store the demoted INT2 history."""
    _require_cuda_tensor(
        recent,
        name="recent",
        ndim=3,
        dtype=torch.bfloat16,
    )
    if prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be positive")
    num_rows = logical_positions.shape[0]
    for name, tensor in (
        ("logical_positions", logical_positions),
        ("hp_rows", hp_rows),
    ):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=1,
            dtype=(torch.int32, torch.int64),
        )
        if tensor.shape[0] != num_rows:
            raise ValueError(f"{name} lengths must match")
    _validate_indices(page_ids, page_offsets, num_rows=num_rows)
    latent_rank = recent.shape[2]
    gathered = torch.empty(
        (num_rows, latent_rank),
        dtype=torch.bfloat16,
        device=recent.device,
    )
    if num_rows:
        _gather_recent_latent_kernel[(num_rows,)](
            recent,
            gathered,
            logical_positions,
            hp_rows,
            num_rows,
            stride_recent_row=recent.stride(0),
            stride_recent_token=recent.stride(1),
            stride_recent_dim=recent.stride(2),
            stride_output_row=gathered.stride(0),
            stride_output_dim=gathered.stride(1),
            prefix_tokens=prefix_tokens,
            recent_tokens=recent.shape[1],
            latent_rank=latent_rank,
            block_d=triton.next_power_of_2(latent_rank),
            num_warps=4,
            num_stages=1,
        )
    return oscar_mla_rotate_quantize_store(
        gathered,
        rotation,
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
        clip_ratio=clip_ratio,
    )


def oscar_mla_dequantize_history(
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize selected history slots to an FP32 oracle tensor."""
    num_groups, group_size, packed_group_bytes, latent_rank = _validate_history_tensors(
        history_data, history_scale, history_zero
    )
    num_rows = page_ids.shape[0]
    _validate_indices(page_ids, page_offsets, num_rows=num_rows)
    if output is None:
        output = torch.empty(
            (num_rows, latent_rank),
            dtype=torch.float32,
            device=history_data.device,
        )
    else:
        _require_cuda_tensor(
            output,
            name="output",
            ndim=2,
            dtype=torch.float32,
        )
        if output.shape != (num_rows, latent_rank):
            raise ValueError("history dequant output shape does not match cache")
    if not (
        history_data.device
        == history_scale.device
        == history_zero.device
        == page_ids.device
        == page_offsets.device
        == output.device
    ):
        raise ValueError("all OSCAR MLA dequant tensors must share one CUDA device")
    if num_rows == 0:
        return output

    _dequantize_history_kernel[(num_rows, num_groups)](
        history_data,
        history_scale,
        history_zero,
        output,
        page_ids,
        page_offsets,
        num_rows,
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        stride_output_row=output.stride(0),
        stride_output_dim=output.stride(1),
        num_groups=num_groups,
        group_size=group_size,
        packed_group_bytes=packed_group_bytes,
        num_warps=4,
        num_stages=1,
    )
    return output
