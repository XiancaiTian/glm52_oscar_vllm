"""Worker-side views and ownership state for OSCAR MLA three-pool caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import OscarMLAAttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.oscar_mla.cache import (
        WorkerCacheMetadata,
    )
    from vllm.v1.core.sched.output import SchedulerOutput


@dataclass(frozen=True)
class OscarMLACacheTensors:
    """Non-overlapping views over one exact per-layer cache allocation."""

    raw: torch.Tensor
    history_data: torch.Tensor
    history_scale: torch.Tensor
    history_zero: torch.Tensor
    prefix: torch.Tensor
    recent: torch.Tensor
    rope: torch.Tensor


def reshape_oscar_mla_cache(
    raw: torch.Tensor,
    spec: OscarMLAAttentionSpec,
    *,
    num_blocks: int,
    max_num_seqs: int,
) -> OscarMLACacheTensors:
    """Carve the raw byte allocation into the three logical cache pools."""
    if raw.dtype != torch.int8 or raw.ndim != 1:
        raise ValueError("OSCAR MLA raw cache must be a flat int8 tensor")
    if num_blocks <= 0 or max_num_seqs <= 0:
        raise ValueError("OSCAR MLA cache dimensions must be positive")

    block_size = spec.block_size
    latent_rank = spec.latent_rank
    num_groups = latent_rank // spec.group_size
    history_slots = num_blocks * block_size
    packed_bytes = latent_rank * 2 // 8

    data_bytes = history_slots * packed_bytes
    metadata_elements = history_slots * num_groups
    metadata_bytes = (
        metadata_elements * torch.tensor([], dtype=torch.float32).element_size()
    )
    prefix_elements = max_num_seqs * spec.prefix_tokens * latent_rank
    recent_elements = max_num_seqs * spec.recent_tokens * latent_rank
    rope_elements = num_blocks * block_size * spec.rope_head_size
    hp_element_size = torch.tensor([], dtype=spec.hp_dtype).element_size()
    expected_bytes = (
        data_bytes
        + 2 * metadata_bytes
        + (prefix_elements + recent_elements + rope_elements) * hp_element_size
    )
    if raw.numel() != expected_bytes:
        raise ValueError(
            "OSCAR MLA raw cache size does not match the planner: "
            f"expected={expected_bytes}, actual={raw.numel()}"
        )

    offset = 0
    history_data = (
        raw.narrow(0, offset, data_bytes)
        .view(torch.uint8)
        .view(num_blocks, block_size, packed_bytes)
    )
    offset += data_bytes
    history_scale = (
        raw.narrow(0, offset, metadata_bytes)
        .view(torch.float32)
        .view(num_blocks, block_size, num_groups)
    )
    offset += metadata_bytes
    history_zero = (
        raw.narrow(0, offset, metadata_bytes)
        .view(torch.float32)
        .view(num_blocks, block_size, num_groups)
    )
    offset += metadata_bytes
    prefix_bytes = prefix_elements * hp_element_size
    prefix = (
        raw.narrow(0, offset, prefix_bytes)
        .view(spec.hp_dtype)
        .view(max_num_seqs, spec.prefix_tokens, latent_rank)
    )
    offset += prefix_bytes
    recent_bytes = recent_elements * hp_element_size
    recent = (
        raw.narrow(0, offset, recent_bytes)
        .view(spec.hp_dtype)
        .view(max_num_seqs, spec.recent_tokens, latent_rank)
    )
    offset += recent_bytes
    rope_bytes = rope_elements * hp_element_size
    rope = (
        raw.narrow(0, offset, rope_bytes)
        .view(spec.hp_dtype)
        .view(num_blocks, block_size, spec.rope_head_size)
    )
    offset += rope_bytes
    assert offset == raw.numel()

    return OscarMLACacheTensors(
        raw=raw,
        history_data=history_data,
        history_scale=history_scale,
        history_zero=history_zero,
        prefix=prefix,
        recent=recent,
        rope=rope,
    )


class OscarMLAWorkerOwnership:
    """Request-keyed worker mirror of scheduler-owned three-pool metadata."""

    def __init__(self) -> None:
        self._metadata: dict[str, WorkerCacheMetadata] = {}

    def apply(self, scheduler_output: SchedulerOutput) -> None:
        released = set(scheduler_output.finished_req_ids)
        if scheduler_output.preempted_req_ids:
            released.update(scheduler_output.preempted_req_ids)
        for request_id in released:
            self._metadata.pop(request_id, None)

        for request_id, metadata in scheduler_output.oscar_mla_cache_metadata.items():
            if metadata.request_id != request_id:
                raise RuntimeError("OSCAR MLA metadata request ID mismatch")
            previous = self._metadata.get(request_id)
            if previous is not None:
                if metadata.generation != previous.generation:
                    raise RuntimeError(
                        "OSCAR MLA generation changed without a release event"
                    )
                if metadata.cache_version < previous.cache_version:
                    raise RuntimeError("stale OSCAR MLA cache metadata")
            self._metadata[request_id] = metadata

    def get(self, request_id: str) -> WorkerCacheMetadata:
        return self._metadata[request_id]

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._metadata
