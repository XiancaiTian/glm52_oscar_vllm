from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.cache import (
    MLACacheCapacityError,
    MLACacheGeometry,
    WorkerCacheMetadata,
    plan_mla_runtime_cache,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import OscarMLAKVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    OscarMLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.oscar_mla_cache import (
    OscarMLAWorkerOwnership,
    reshape_oscar_mla_cache,
)


def _spec() -> OscarMLAAttentionSpec:
    return OscarMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        cache_dtype_str="oscar_mla_int2",
        latent_rank=512,
        rope_head_size=64,
        history_slot_size=160,
        group_size=128,
        prefix_tokens=64,
        recent_tokens=256,
    )


def test_oscar_mla_spec_accounts_only_history_pages() -> None:
    spec = _spec()

    assert spec.page_size_bytes == 16 * (160 + 64 * 2)
    assert spec.bf16_token_size_bytes == 512 * 2
    assert OscarMLAAttentionSpec.merge([spec, spec]) == spec

    with pytest.raises(ValueError, match="expected=160"):
        OscarMLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="oscar_mla_int2",
            latent_rank=512,
            rope_head_size=64,
            history_slot_size=159,
        )


def test_cache_config_matches_three_pool_plan_exactly() -> None:
    num_layers = 78
    max_num_seqs = 16
    available_memory = 14 * 1024**3
    main_layer_names = [
        f"model.layers.{i}.self_attn.mla_attn" for i in range(num_layers)
    ]
    index_layer_names = [
        f"model.layers.{i}.self_attn.indexer.k_cache"
        for i in (0, 1, 2, *range(6, 78, 4))
    ]
    per_layer_specs = {layer_name: _spec() for layer_name in main_layer_names}
    per_layer_specs.update(
        {
            layer_name: MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
            )
            for layer_name in index_layer_names
        }
    )
    layer_names = main_layer_names + index_layer_names
    groups = [
        KVCacheGroupSpec(
            layer_names=layer_names,
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs=per_layer_specs,
            ),
        )
    ]
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )

    config = get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory,
        suppress_log=True,
    )
    expected = plan_mla_runtime_cache(
        MLACacheGeometry(num_layers=num_layers, latent_rank=512),
        total_memory_bytes=available_memory,
        max_num_seqs=max_num_seqs,
        rope_bytes_per_layer_token=64 * 2,
        auxiliary_bytes_per_block=21 * 16 * 132,
    )

    assert config.num_blocks == expected.num_blocks
    assert config.oscar_mla_history_pages == expected.history_pages
    assert config.oscar_mla_max_num_seqs == max_num_seqs
    assert len(config.kv_cache_tensors) == len(layer_names)
    assert sum(tensor.size for tensor in config.kv_cache_tensors) == (
        expected.allocated_bytes
    )
    assert all(len(tensor.shared_by) == 1 for tensor in config.kv_cache_tensors)
    vllm_config.model_config = SimpleNamespace(max_model_len=32768)
    assert get_max_concurrency_for_kv_cache_config(vllm_config, config) == 16
    scheduler_config = generate_scheduler_kv_cache_config([config])
    assert isinstance(
        scheduler_config.kv_cache_groups[0].kv_cache_spec,
        OscarMLAAttentionSpec,
    )


def test_worker_views_cover_raw_allocation_without_padding() -> None:
    spec = _spec()
    num_blocks = 3
    max_num_seqs = 2
    history_slots = num_blocks * spec.block_size
    raw_bytes = (
        history_slots * spec.history_slot_size
        + max_num_seqs
        * (spec.prefix_tokens + spec.recent_tokens)
        * spec.bf16_token_size_bytes
        + history_slots * spec.rope_head_size * 2
    )
    raw = torch.empty(raw_bytes, dtype=torch.int8)

    tensors = reshape_oscar_mla_cache(
        raw,
        spec,
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
    )

    assert tensors.history_data.shape == (3, 16, 128)
    assert tensors.history_scale.shape == (3, 16, 4)
    assert tensors.history_zero.shape == (3, 16, 4)
    assert tensors.prefix.shape == (2, 64, 512)
    assert tensors.recent.shape == (2, 256, 512)
    assert tensors.rope.shape == (3, 16, 64)
    data_bytes = history_slots * 128
    metadata_bytes = history_slots * 4 * 4
    prefix_bytes = max_num_seqs * spec.prefix_tokens * 512 * 2
    assert tensors.history_data.data_ptr() == raw.data_ptr()
    assert tensors.history_scale.data_ptr() == raw.data_ptr() + data_bytes
    assert tensors.history_zero.data_ptr() == (
        raw.data_ptr() + data_bytes + metadata_bytes
    )
    assert tensors.prefix.data_ptr() == (
        raw.data_ptr() + data_bytes + 2 * metadata_bytes
    )
    assert tensors.recent.data_ptr() == (
        raw.data_ptr() + data_bytes + 2 * metadata_bytes + prefix_bytes
    )
    recent_bytes = max_num_seqs * spec.recent_tokens * 512 * 2
    assert tensors.rope.data_ptr() == (
        raw.data_ptr() + data_bytes + 2 * metadata_bytes + prefix_bytes + recent_bytes
    )
    assert tensors.recent.untyped_storage().nbytes() == raw.numel()


def test_scheduler_manager_allocates_history_only_and_reuses_generation() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=40,
        enable_caching=False,
        hash_block_size=16,
    )
    manager = OscarMLAKVCacheManager(
        _spec(),
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        max_num_seqs=1,
        history_pages=2,
    )

    assert len(manager.allocate_new_blocks("r0", 320, 320)) == 20
    initial = manager.metadata("r0")
    assert initial.hp_row == 0
    assert initial.history_pages == ()

    first = manager.allocate_new_blocks("r0", 321, 321)
    assert len(first) == 1
    assert manager.metadata("r0").partial_history_slots == 1
    second = manager.allocate_new_blocks("r0", 337, 337)
    assert len(second) == 1
    before_oom = manager.metadata("r0")

    standard_blocks_before_oom = len(manager.req_to_blocks["r0"])
    with pytest.raises(MLACacheCapacityError):
        manager.allocate_new_blocks("r0", 353, 353)

    assert manager.metadata("r0") == before_oom
    assert len(manager.req_to_blocks["r0"]) == standard_blocks_before_oom
    first_generation = before_oom.generation
    manager.free("r0")
    manager.allocate_new_blocks("reused", 64, 64)
    reused = manager.metadata("reused")
    assert reused.hp_row == 0
    assert reused.generation != first_generation


def test_kv_cache_manager_exposes_scheduler_metadata() -> None:
    config = KVCacheConfig(
        num_blocks=30,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=_spec())
        ],
        oscar_mla_max_num_seqs=1,
        oscar_mla_history_pages=3,
    )
    cache_manager = KVCacheManager(
        config,
        max_model_len=32768,
        hash_block_size=16,
        enable_caching=False,
    )
    runtime_manager = cache_manager.coordinator.single_type_managers[0]
    assert isinstance(runtime_manager, OscarMLAKVCacheManager)
    runtime_manager.allocate_new_blocks("r0", 321, 321)

    metadata = cache_manager.get_oscar_mla_metadata(["r0"])

    assert metadata["r0"].logical_length == 321
    assert len(metadata["r0"].history_pages) == 1


def test_worker_ownership_rejects_stale_versions_and_handles_reuse() -> None:
    ownership = OscarMLAWorkerOwnership()
    output = SchedulerOutput.make_empty()
    current = WorkerCacheMetadata(
        request_id="r0",
        generation=1,
        cache_version=2,
        logical_length=337,
        hp_row=0,
        prefix_start=0,
        recent_start=0,
        history_pages=(1, 2),
        partial_history_slots=1,
    )
    output.oscar_mla_cache_metadata = {"r0": current}
    ownership.apply(output)
    assert ownership.get("r0") == current

    output.oscar_mla_cache_metadata = {
        "r0": WorkerCacheMetadata(**{**current.__dict__, "cache_version": 1})
    }
    with pytest.raises(RuntimeError, match="stale"):
        ownership.apply(output)

    output.finished_req_ids = {"r0"}
    output.oscar_mla_cache_metadata = {
        "r0": WorkerCacheMetadata(
            **{**current.__dict__, "generation": 2, "cache_version": 1}
        )
    }
    ownership.apply(output)
    assert ownership.get("r0").generation == 2
