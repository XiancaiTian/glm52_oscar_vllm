"""CPU capacity and ownership model for three-pool OSCAR MLA caches."""

from __future__ import annotations

from dataclasses import dataclass, field


class MLACacheCapacityError(RuntimeError):
    """Raised when a three-pool cache allocation cannot be satisfied."""


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class MLACacheGeometry:
    num_layers: int
    latent_rank: int
    group_size: int = 128
    block_size: int = 16
    prefix_tokens: int = 64
    recent_tokens: int = 256
    quant_bits: int = 2
    metadata_element_bytes: int = 4
    bf16_element_bytes: int = 2

    def __post_init__(self) -> None:
        positive = {
            "num_layers": self.num_layers,
            "latent_rank": self.latent_rank,
            "group_size": self.group_size,
            "block_size": self.block_size,
            "metadata_element_bytes": self.metadata_element_bytes,
            "bf16_element_bytes": self.bf16_element_bytes,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"MLA cache geometry values must be positive: {invalid}")
        if self.quant_bits != 2:
            raise ValueError("OSCAR MLA history currently requires INT2")
        if self.latent_rank % self.group_size:
            raise ValueError("group_size must exactly divide latent_rank")
        if self.latent_rank * self.quant_bits % 8:
            raise ValueError("INT2 latent rows must pack into whole bytes")
        if self.prefix_tokens < 0 or self.recent_tokens < 0:
            raise ValueError("prefix and recent windows must be non-negative")
        if self.prefix_tokens % self.block_size:
            raise ValueError("prefix window must be block aligned")
        if self.recent_tokens % self.block_size:
            raise ValueError("recent window must be block aligned")

    @property
    def num_groups(self) -> int:
        return self.latent_rank // self.group_size

    @property
    def history_data_bytes_per_layer_token(self) -> int:
        return self.latent_rank * self.quant_bits // 8

    @property
    def history_metadata_bytes_per_layer_token(self) -> int:
        scale_and_zero_point = 2
        return self.num_groups * scale_and_zero_point * self.metadata_element_bytes

    @property
    def history_bytes_per_layer_token(self) -> int:
        return (
            self.history_data_bytes_per_layer_token
            + self.history_metadata_bytes_per_layer_token
        )

    @property
    def history_token_bytes(self) -> int:
        return self.num_layers * self.history_bytes_per_layer_token

    @property
    def bf16_token_bytes(self) -> int:
        return self.num_layers * self.latent_rank * self.bf16_element_bytes

    @property
    def history_page_bytes(self) -> int:
        return self.block_size * self.history_token_bytes

    @property
    def bf16_page_bytes(self) -> int:
        return self.block_size * self.bf16_token_bytes

    def partition(self, sequence_length: int) -> TokenPartition:
        if sequence_length < 0:
            raise ValueError("sequence_length must be non-negative")
        prefix = min(sequence_length, self.prefix_tokens)
        history = max(
            0,
            sequence_length - self.prefix_tokens - self.recent_tokens,
        )
        recent = sequence_length - prefix - history
        return TokenPartition(prefix=prefix, recent=recent, history=history)


@dataclass(frozen=True)
class TokenPartition:
    prefix: int
    recent: int
    history: int

    @property
    def total(self) -> int:
        return self.prefix + self.recent + self.history


@dataclass(frozen=True)
class MLACachePlan:
    geometry: MLACacheGeometry
    total_memory_bytes: int
    max_num_seqs: int
    prefix_slots: int
    recent_slots: int
    history_pages: int
    unused_bytes: int

    @property
    def history_slots(self) -> int:
        return self.history_pages * self.geometry.block_size

    @property
    def history_bytes(self) -> int:
        return self.history_slots * self.geometry.history_token_bytes

    @property
    def bf16_bytes(self) -> int:
        return (self.prefix_slots + self.recent_slots) * self.geometry.bf16_token_bytes

    @property
    def allocated_bytes(self) -> int:
        return self.bf16_bytes + self.history_bytes

    @property
    def native_bf16_slots(self) -> int:
        pages = self.total_memory_bytes // self.geometry.bf16_page_bytes
        return pages * self.geometry.block_size

    @property
    def theoretical_history_compression_ratio(self) -> float:
        return self.geometry.bf16_token_bytes / self.geometry.history_token_bytes

    @property
    def padded_history_compression_ratio(self) -> float:
        return self.geometry.bf16_page_bytes / self.geometry.history_page_bytes

    @property
    def allocated_capacity_ratio(self) -> float:
        slots = self.prefix_slots + self.recent_slots + self.history_slots
        return slots / self.native_bf16_slots

    @property
    def guaranteed_history_slots(self) -> int:
        fragmentation = self.max_num_seqs * (self.geometry.block_size - 1)
        return max(0, self.history_slots - fragmentation)

    @property
    def guaranteed_capacity_ratio(self) -> float:
        slots = self.prefix_slots + self.recent_slots + self.guaranteed_history_slots
        return slots / self.native_bf16_slots


def plan_mla_cache(
    geometry: MLACacheGeometry,
    *,
    total_memory_bytes: int,
    max_num_seqs: int,
) -> MLACachePlan:
    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be positive")
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    prefix_slots = max_num_seqs * geometry.prefix_tokens
    recent_slots = max_num_seqs * geometry.recent_tokens
    bf16_bytes = (prefix_slots + recent_slots) * geometry.bf16_token_bytes
    if bf16_bytes >= total_memory_bytes:
        raise MLACacheCapacityError(
            "BF16 prefix/recent pools consume the cache budget: "
            f"required={bf16_bytes}, budget={total_memory_bytes}"
        )
    history_pages = (total_memory_bytes - bf16_bytes) // geometry.history_page_bytes
    if history_pages <= 0:
        raise MLACacheCapacityError("cache budget has no room for INT2 history")
    allocated = bf16_bytes + history_pages * geometry.history_page_bytes
    return MLACachePlan(
        geometry=geometry,
        total_memory_bytes=total_memory_bytes,
        max_num_seqs=max_num_seqs,
        prefix_slots=prefix_slots,
        recent_slots=recent_slots,
        history_pages=history_pages,
        unused_bytes=total_memory_bytes - allocated,
    )


@dataclass(frozen=True)
class MLARuntimeCachePlan:
    """Joint budget for latent pools and uncompressed auxiliary caches."""

    geometry: MLACacheGeometry
    total_memory_bytes: int
    max_num_seqs: int
    num_blocks: int
    rope_bytes_per_layer_token: int
    auxiliary_bytes_per_block: int
    unused_bytes: int

    @property
    def usable_blocks(self) -> int:
        # vLLM reserves block ID 0 as its null block.
        return self.num_blocks - 1

    @property
    def logical_token_slots(self) -> int:
        return self.usable_blocks * self.geometry.block_size

    @property
    def history_pages(self) -> int:
        # History has an independent page namespace and no null block.
        return self.num_blocks

    @property
    def fixed_prefix_slots(self) -> int:
        return self.max_num_seqs * self.geometry.prefix_tokens

    @property
    def fixed_recent_slots(self) -> int:
        return self.max_num_seqs * self.geometry.recent_tokens

    @property
    def history_slots(self) -> int:
        return self.history_pages * self.geometry.block_size

    @property
    def fixed_prefix_bytes(self) -> int:
        return self.fixed_prefix_slots * self.geometry.bf16_token_bytes

    @property
    def fixed_recent_bytes(self) -> int:
        return self.fixed_recent_slots * self.geometry.bf16_token_bytes

    @property
    def fixed_bf16_bytes(self) -> int:
        return self.fixed_prefix_bytes + self.fixed_recent_bytes

    @property
    def history_bytes(self) -> int:
        return self.history_pages * self.geometry.history_page_bytes

    @property
    def rope_bytes(self) -> int:
        return (
            self.num_blocks
            * self.geometry.block_size
            * self.geometry.num_layers
            * self.rope_bytes_per_layer_token
        )

    @property
    def auxiliary_bytes(self) -> int:
        return self.num_blocks * self.auxiliary_bytes_per_block

    @property
    def allocated_bytes(self) -> int:
        return (
            self.fixed_bf16_bytes
            + self.history_bytes
            + self.rope_bytes
            + self.auxiliary_bytes
        )

    @property
    def theoretical_history_compression_ratio(self) -> float:
        return self.geometry.bf16_token_bytes / self.geometry.history_token_bytes

    @property
    def padded_history_compression_ratio(self) -> float:
        return self.geometry.bf16_page_bytes / self.geometry.history_page_bytes

    @property
    def native_page_bytes(self) -> int:
        return (
            self.geometry.bf16_page_bytes
            + self.geometry.num_layers
            * self.geometry.block_size
            * self.rope_bytes_per_layer_token
            + self.auxiliary_bytes_per_block
        )

    @property
    def native_num_blocks(self) -> int:
        return self.total_memory_bytes // self.native_page_bytes

    @property
    def native_logical_token_slots(self) -> int:
        usable_blocks = max(0, self.native_num_blocks - 1)
        return usable_blocks * self.geometry.block_size

    @property
    def allocated_capacity_ratio(self) -> float:
        if self.native_logical_token_slots == 0:
            return float("inf")
        return self.logical_token_slots / self.native_logical_token_slots


def plan_mla_runtime_cache(
    geometry: MLACacheGeometry,
    *,
    total_memory_bytes: int,
    max_num_seqs: int,
    rope_bytes_per_layer_token: int,
    auxiliary_bytes_per_block: int,
) -> MLARuntimeCachePlan:
    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be positive")
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    if rope_bytes_per_layer_token <= 0:
        raise ValueError("rope_bytes_per_layer_token must be positive")
    if auxiliary_bytes_per_block < 0:
        raise ValueError("auxiliary_bytes_per_block must be non-negative")

    fixed_slots = max_num_seqs * (geometry.prefix_tokens + geometry.recent_tokens)
    fixed_bf16_bytes = fixed_slots * geometry.bf16_token_bytes
    if fixed_bf16_bytes >= total_memory_bytes:
        raise MLACacheCapacityError(
            "BF16 prefix/recent pools consume the cache budget: "
            f"required={fixed_bf16_bytes}, budget={total_memory_bytes}"
        )
    variable_bytes_per_block = (
        geometry.history_page_bytes
        + geometry.num_layers * geometry.block_size * rope_bytes_per_layer_token
        + auxiliary_bytes_per_block
    )
    num_blocks = (total_memory_bytes - fixed_bf16_bytes) // variable_bytes_per_block
    if num_blocks <= 1:
        raise MLACacheCapacityError(
            "cache budget has no usable block after the vLLM null block"
        )
    allocated = fixed_bf16_bytes + num_blocks * variable_bytes_per_block
    return MLARuntimeCachePlan(
        geometry=geometry,
        total_memory_bytes=total_memory_bytes,
        max_num_seqs=max_num_seqs,
        num_blocks=num_blocks,
        rope_bytes_per_layer_token=rope_bytes_per_layer_token,
        auxiliary_bytes_per_block=auxiliary_bytes_per_block,
        unused_bytes=total_memory_bytes - allocated,
    )


@dataclass
class _IndexPool:
    capacity: int
    free: list[int] = field(init=False)
    allocated: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("pool capacity must be positive")
        self.free = list(range(self.capacity - 1, -1, -1))

    def allocate(self, count: int = 1) -> tuple[int, ...]:
        if count < 0:
            raise ValueError("allocation count must be non-negative")
        if count > len(self.free):
            raise MLACacheCapacityError(
                f"requested {count} slots with {len(self.free)} free"
            )
        result = tuple(self.free.pop() for _ in range(count))
        self.allocated.update(result)
        return result

    def release(self, values: tuple[int, ...] | list[int]) -> None:
        unique = set(values)
        if len(unique) != len(values):
            raise RuntimeError("duplicate value in pool release")
        if not unique.issubset(self.allocated):
            raise RuntimeError(
                f"release of unallocated values: {sorted(unique - self.allocated)}"
            )
        self.allocated.difference_update(unique)
        self.free.extend(unique)

    def assert_consistent(self) -> None:
        free = set(self.free)
        if len(free) != len(self.free):
            raise RuntimeError("pool contains duplicate free values")
        if free & self.allocated:
            raise RuntimeError("pool value is both free and allocated")
        if free | self.allocated != set(range(self.capacity)):
            raise RuntimeError("pool accounting is not conserved")


@dataclass
class RequestOwnership:
    request_id: str
    generation: int
    hp_row: int
    prefix_start: int
    recent_start: int
    logical_length: int = 0
    cache_version: int = 0
    history_tokens: int = 0
    full_history_pages: list[int] = field(default_factory=list)
    partial_history_page: int | None = None
    partial_history_slots: int = 0

    @property
    def history_pages(self) -> tuple[int, ...]:
        pages = list(self.full_history_pages)
        if self.partial_history_page is not None:
            pages.append(self.partial_history_page)
        return tuple(pages)


@dataclass(frozen=True)
class LengthUpdate:
    previous_length: int
    new_length: int
    demote_start: int
    demote_end: int
    cache_version: int


@dataclass(frozen=True)
class WorkerCacheMetadata:
    request_id: str
    generation: int
    cache_version: int
    logical_length: int
    hp_row: int
    prefix_start: int
    recent_start: int
    history_pages: tuple[int, ...]
    partial_history_slots: int


class MLATriPoolAllocator:
    """Stable request ownership and incremental INT2 history allocation."""

    def __init__(self, plan: MLACachePlan) -> None:
        self.plan = plan
        self._rows = _IndexPool(plan.max_num_seqs)
        self._history = _IndexPool(plan.history_pages)
        self._next_generation = 1
        self.requests: dict[str, RequestOwnership] = {}

    def required_history_pages(self, request_id: str, new_length: int) -> int:
        request = self.requests.get(request_id)
        current_length = request.logical_length if request is not None else 0
        current_history = request.history_tokens if request is not None else 0
        partial_slots = request.partial_history_slots if request is not None else 0
        if new_length < current_length:
            raise ValueError("logical length must not decrease")
        target_history = self.plan.geometry.partition(new_length).history
        new_history = target_history - current_history
        available = (
            self.plan.geometry.block_size - partial_slots
            if request is not None and request.partial_history_page is not None
            else 0
        )
        return _ceil_div(
            max(0, new_history - available),
            self.plan.geometry.block_size,
        )

    def can_update_length(self, request_id: str, new_length: int) -> bool:
        if request_id not in self.requests and not self._rows.free:
            return False
        return self.required_history_pages(request_id, new_length) <= len(
            self._history.free
        )

    def start_request(self, request_id: str) -> RequestOwnership:
        if request_id in self.requests:
            raise RuntimeError(f"duplicate request: {request_id}")
        row = self._rows.allocate()[0]
        generation = self._next_generation
        self._next_generation += 1
        ownership = RequestOwnership(
            request_id=request_id,
            generation=generation,
            hp_row=row,
            prefix_start=row * self.plan.geometry.prefix_tokens,
            recent_start=row * self.plan.geometry.recent_tokens,
        )
        self.requests[request_id] = ownership
        return ownership

    def update_length(self, request_id: str, new_length: int) -> LengthUpdate:
        request = self.requests[request_id]
        if new_length < request.logical_length:
            raise ValueError("logical length must not decrease")
        target_history = self.plan.geometry.partition(new_length).history
        new_history = target_history - request.history_tokens
        block_size = self.plan.geometry.block_size
        available = (
            block_size - request.partial_history_slots
            if request.partial_history_page is not None
            else 0
        )
        pages_needed = self.required_history_pages(request_id, new_length)
        new_pages = self._history.allocate(pages_needed)

        remaining = new_history
        if request.partial_history_page is not None:
            consumed = min(remaining, available)
            request.partial_history_slots += consumed
            remaining -= consumed
            if request.partial_history_slots == block_size:
                request.full_history_pages.append(request.partial_history_page)
                request.partial_history_page = None
                request.partial_history_slots = 0

        page_index = 0
        while remaining >= block_size:
            request.full_history_pages.append(new_pages[page_index])
            page_index += 1
            remaining -= block_size
        if remaining:
            request.partial_history_page = new_pages[page_index]
            request.partial_history_slots = remaining

        previous_length = request.logical_length
        demote_start = self.plan.geometry.prefix_tokens + request.history_tokens
        request.logical_length = new_length
        request.history_tokens = target_history
        request.cache_version += 1
        return LengthUpdate(
            previous_length=previous_length,
            new_length=new_length,
            demote_start=demote_start,
            demote_end=demote_start + new_history,
            cache_version=request.cache_version,
        )

    def metadata(
        self,
        request_id: str,
        *,
        expected_generation: int | None = None,
    ) -> WorkerCacheMetadata:
        request = self.requests[request_id]
        if (
            expected_generation is not None
            and request.generation != expected_generation
        ):
            raise RuntimeError(
                f"stale cache generation {expected_generation}; "
                f"current={request.generation}"
            )
        return WorkerCacheMetadata(
            request_id=request.request_id,
            generation=request.generation,
            cache_version=request.cache_version,
            logical_length=request.logical_length,
            hp_row=request.hp_row,
            prefix_start=request.prefix_start,
            recent_start=request.recent_start,
            history_pages=request.history_pages,
            partial_history_slots=request.partial_history_slots,
        )

    def prefix_slot(self, request_id: str, token_position: int) -> int:
        request = self.requests[request_id]
        if (
            not 0
            <= token_position
            < min(
                request.logical_length,
                self.plan.geometry.prefix_tokens,
            )
        ):
            raise ValueError("token is not in the request prefix")
        return request.prefix_start + token_position

    def recent_slot(self, request_id: str, token_position: int) -> int:
        request = self.requests[request_id]
        geometry = self.plan.geometry
        recent_begin = max(
            geometry.prefix_tokens,
            request.logical_length - geometry.recent_tokens,
        )
        if not recent_begin <= token_position < request.logical_length:
            raise ValueError("token is not in the request recent window")
        offset = (token_position - geometry.prefix_tokens) % geometry.recent_tokens
        return request.recent_start + offset

    def history_slot(self, request_id: str, token_position: int) -> tuple[int, int]:
        request = self.requests[request_id]
        geometry = self.plan.geometry
        history_offset = token_position - geometry.prefix_tokens
        if not 0 <= history_offset < request.history_tokens:
            raise ValueError("token is not in the request history")
        page_index, slot = divmod(history_offset, geometry.block_size)
        return request.history_pages[page_index], slot

    def finish_request(self, request_id: str) -> None:
        self._release_request(request_id)

    def abort_request(self, request_id: str) -> None:
        self._release_request(request_id)

    def preempt_request(self, request_id: str) -> None:
        self._release_request(request_id)

    def _release_request(self, request_id: str) -> None:
        request = self.requests.pop(request_id)
        self._history.release(list(request.history_pages))
        self._rows.release([request.hp_row])

    def assert_consistent(self) -> None:
        self._rows.assert_consistent()
        self._history.assert_consistent()
        owned_rows = {request.hp_row for request in self.requests.values()}
        if owned_rows != self._rows.allocated:
            raise RuntimeError("request HP row ownership is inconsistent")
        owned_pages: set[int] = set()
        block_size = self.plan.geometry.block_size
        for request in self.requests.values():
            pages = request.history_pages
            if owned_pages & set(pages):
                raise RuntimeError("history page is owned by multiple requests")
            owned_pages.update(pages)
            if not 0 <= request.partial_history_slots < block_size:
                raise RuntimeError("invalid partial history occupancy")
            if (request.partial_history_page is None) != (
                request.partial_history_slots == 0
            ):
                raise RuntimeError("partial history state is inconsistent")
            if len(pages) != _ceil_div(request.history_tokens, block_size):
                raise RuntimeError("history page count does not match token count")
            partition = self.plan.geometry.partition(request.logical_length)
            if partition.history != request.history_tokens:
                raise RuntimeError("history tokens do not match logical length")
        if owned_pages != self._history.allocated:
            raise RuntimeError("request history ownership is inconsistent")
