import pytest

from vllm.model_executor.layers.quantization.oscar_mla.cache import (
    MLACacheCapacityError,
    MLACacheGeometry,
    MLATriPoolAllocator,
    plan_mla_cache,
)


def _geometry() -> MLACacheGeometry:
    return MLACacheGeometry(num_layers=78, latent_rank=512)


def test_glm52_geometry_has_exact_int2_and_bf16_bytes() -> None:
    geometry = _geometry()

    assert geometry.num_groups == 4
    assert geometry.history_data_bytes_per_layer_token == 128
    assert geometry.history_metadata_bytes_per_layer_token == 32
    assert geometry.history_bytes_per_layer_token == 160
    assert geometry.history_token_bytes == 12480
    assert geometry.bf16_token_bytes == 79872
    assert geometry.bf16_token_bytes / geometry.history_token_bytes == 6.4


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (0, (0, 0, 0)),
        (63, (63, 0, 0)),
        (64, (64, 0, 0)),
        (65, (64, 1, 0)),
        (319, (64, 255, 0)),
        (320, (64, 256, 0)),
        (321, (64, 256, 1)),
        (32768, (64, 256, 32448)),
    ],
)
def test_token_partition_boundaries(length, expected) -> None:
    partition = _geometry().partition(length)

    assert (partition.prefix, partition.recent, partition.history) == expected
    assert partition.total == length


def test_capacity_plan_conserves_bytes_and_reports_mixed_ratios() -> None:
    geometry = _geometry()
    plan = plan_mla_cache(
        geometry,
        total_memory_bytes=14 * 1024**3,
        max_num_seqs=16,
    )

    assert plan.prefix_slots == 1024
    assert plan.recent_slots == 4096
    assert plan.allocated_bytes + plan.unused_bytes == plan.total_memory_bytes
    assert 0 <= plan.unused_bytes < geometry.history_page_bytes
    assert plan.theoretical_history_compression_ratio == pytest.approx(6.4)
    assert plan.padded_history_compression_ratio == pytest.approx(6.4)
    assert plan.guaranteed_capacity_ratio <= plan.allocated_capacity_ratio
    assert plan.guaranteed_capacity_ratio > 5


def test_capacity_plan_rejects_bf16_window_exhaustion() -> None:
    with pytest.raises(MLACacheCapacityError, match="consume"):
        plan_mla_cache(
            _geometry(),
            total_memory_bytes=1024,
            max_num_seqs=16,
        )


def test_request_ranges_are_stable_and_generation_changes_on_reuse() -> None:
    plan = plan_mla_cache(
        _geometry(),
        total_memory_bytes=2 * 1024**3,
        max_num_seqs=2,
    )
    allocator = MLATriPoolAllocator(plan)

    first = allocator.start_request("first")
    second = allocator.start_request("second")
    assert (first.hp_row, first.prefix_start, first.recent_start) == (0, 0, 0)
    assert (second.hp_row, second.prefix_start, second.recent_start) == (
        1,
        64,
        256,
    )
    old_generation = first.generation
    allocator.finish_request("first")
    reused = allocator.start_request("reused")
    assert reused.hp_row == 0
    assert reused.generation != old_generation
    with pytest.raises(RuntimeError, match="stale"):
        allocator.metadata("reused", expected_generation=old_generation)
    allocator.assert_consistent()


def test_incremental_history_pages_and_physical_addresses() -> None:
    plan = plan_mla_cache(
        _geometry(),
        total_memory_bytes=2 * 1024**3,
        max_num_seqs=1,
    )
    allocator = MLATriPoolAllocator(plan)
    request = allocator.start_request("r0")

    no_demotion = allocator.update_length("r0", 320)
    assert (no_demotion.demote_start, no_demotion.demote_end) == (64, 64)
    update = allocator.update_length("r0", 337)
    assert (update.demote_start, update.demote_end) == (64, 81)
    assert request.history_tokens == 17
    assert len(request.full_history_pages) == 1
    assert request.partial_history_slots == 1
    first_page, second_page = request.history_pages
    assert allocator.history_slot("r0", 64) == (first_page, 0)
    assert allocator.history_slot("r0", 79) == (first_page, 15)
    assert allocator.history_slot("r0", 80) == (second_page, 0)
    assert allocator.prefix_slot("r0", 63) == 63
    assert allocator.recent_slot("r0", 81) == 17
    assert allocator.recent_slot("r0", 336) == 16
    allocator.assert_consistent()


def test_history_oom_rolls_back_length_pages_and_version() -> None:
    geometry = _geometry()
    bf16_bytes = (
        geometry.prefix_tokens + geometry.recent_tokens
    ) * geometry.bf16_token_bytes
    plan = plan_mla_cache(
        geometry,
        total_memory_bytes=bf16_bytes + geometry.history_page_bytes,
        max_num_seqs=1,
    )
    allocator = MLATriPoolAllocator(plan)
    request = allocator.start_request("r0")
    allocator.update_length("r0", 336)
    before = allocator.metadata("r0")

    with pytest.raises(MLACacheCapacityError):
        allocator.update_length("r0", 337)

    after = allocator.metadata("r0")
    assert after == before
    assert request.logical_length == 336
    allocator.assert_consistent()


@pytest.mark.parametrize(
    "release_method",
    ["finish_request", "abort_request", "preempt_request"],
)
def test_all_release_paths_conserve_capacity(release_method) -> None:
    plan = plan_mla_cache(
        _geometry(),
        total_memory_bytes=2 * 1024**3,
        max_num_seqs=1,
    )
    allocator = MLATriPoolAllocator(plan)
    allocator.start_request("r0")
    allocator.update_length("r0", 1024)

    getattr(allocator, release_method)("r0")

    allocator.assert_consistent()
    assert not allocator.requests
    assert not allocator._rows.allocated
    assert not allocator._history.allocated


def test_length_cannot_move_backwards() -> None:
    plan = plan_mla_cache(
        _geometry(),
        total_memory_bytes=2 * 1024**3,
        max_num_seqs=1,
    )
    allocator = MLATriPoolAllocator(plan)
    allocator.start_request("r0")
    allocator.update_length("r0", 400)

    with pytest.raises(ValueError, match="must not decrease"):
        allocator.update_length("r0", 399)
