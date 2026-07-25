from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla import triton_mla_sparse
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseImpl,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseMetadata,
)
from vllm.v1.worker.oscar_mla_cache import (
    OscarMLABatchMetadata,
    OscarMLACacheTensors,
)


def _cache() -> OscarMLACacheTensors:
    return OscarMLACacheTensors(
        raw=torch.empty(0, dtype=torch.int8),
        history_data=torch.zeros(12, 16, 128, dtype=torch.uint8),
        history_scale=torch.zeros(12, 16, 4),
        history_zero=torch.zeros(12, 16, 4),
        prefix=torch.zeros(3, 64, 512, dtype=torch.bfloat16),
        recent=torch.zeros(3, 256, 512, dtype=torch.bfloat16),
        rope=torch.zeros(32, 16, 64, dtype=torch.bfloat16),
    )


def _metadata(
    *,
    query_start: int,
    seq_len: int,
    num_tokens: int,
    demote: bool,
) -> XPUMLASparseMetadata:
    demotion_positions = (
        torch.arange(64, 81, dtype=torch.int32)
        if demote
        else torch.empty(0, dtype=torch.int32)
    )
    oscar = OscarMLABatchMetadata(
        hp_rows=torch.tensor([2], dtype=torch.int32),
        history_page_table=torch.tensor([[9, 11]], dtype=torch.int32),
        previous_seq_lens=torch.tensor(
            [320 if demote else 0],
            dtype=torch.int32,
        ),
        demotion_request_indices=torch.zeros(
            demotion_positions.numel(),
            dtype=torch.int32,
        ),
        demotion_positions=demotion_positions,
        demotion_page_ids=torch.tensor(
            [9] * 16 + [11] if demote else [],
            dtype=torch.int32,
        ),
        demotion_page_offsets=torch.tensor(
            list(range(16)) + [0] if demote else [],
            dtype=torch.int32,
        ),
    )
    return XPUMLASparseMetadata(
        num_reqs=1,
        max_query_len=num_tokens,
        max_seq_len=seq_len,
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
        block_table=torch.arange(32, dtype=torch.int32).unsqueeze(0),
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        oscar_mla=oscar,
        block_size=16,
        base_seq_len=query_start,
    )


def _impl() -> TritonMLASparseImpl:
    impl = object.__new__(TritonMLASparseImpl)
    impl.oscar_write_calls = 0
    impl.oscar_demotion_calls = 0
    impl.oscar_read_calls = 0
    return impl


def test_runtime_write_demotes_before_overwriting_recent(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_rope",
        lambda *args: events.append(("rope", args[2].clone())),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        lambda *args, **kwargs: events.append(("demote", args[5].clone())),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        lambda *args, **kwargs: events.append(("history", args[0].shape[0])),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_bf16",
        lambda *args: events.append(("bf16", args[3].clone())),
    )
    metadata = _metadata(
        query_start=320,
        seq_len=337,
        num_tokens=17,
        demote=True,
    )

    impl = _impl()
    impl.do_oscar_kv_cache_update(
        torch.randn(17, 512, dtype=torch.bfloat16),
        torch.randn(17, 1, 64, dtype=torch.bfloat16),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert [event[0] for event in events] == [
        "rope",
        "demote",
        "history",
        "bf16",
    ]
    assert torch.equal(events[1][1], torch.arange(64, 81, dtype=torch.int32))
    assert events[2][1] == 0
    assert torch.equal(events[3][1], torch.arange(320, 337, dtype=torch.int32))
    assert impl.oscar_write_calls == 1
    assert impl.oscar_demotion_calls == 1


def test_runtime_write_directly_stores_current_history(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        lambda *args, **kwargs: None,
    )

    def _capture_history(*args, **kwargs) -> None:
        captured["latent"] = args[0]
        captured["pages"] = args[5]
        captured["offsets"] = args[6]

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        _capture_history,
    )
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_bf16", lambda *args: None)
    metadata = _metadata(
        query_start=0,
        seq_len=337,
        num_tokens=337,
        demote=False,
    )

    impl = _impl()
    impl.do_oscar_kv_cache_update(
        torch.arange(337 * 512, dtype=torch.float32).view(337, 512),
        torch.zeros(337, 1, 64),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert captured["latent"].shape == (17, 512)
    assert captured["pages"].tolist() == [9] * 16 + [11]
    assert captured["offsets"].tolist() == list(range(16)) + [0]


def test_runtime_read_uses_local_dsa_ids_and_three_pool_cache(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}

    def _read(*args, **kwargs):
        captured["selected"] = args[2]
        captured["query_positions"] = args[4]
        captured["block_table"] = args[8]
        return torch.ones(1, 2, 512), torch.zeros(1, 2)

    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_sparse_prefill", _read)
    metadata = _metadata(
        query_start=320,
        seq_len=321,
        num_tokens=1,
        demote=False,
    )
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.softmax_scale = 576**-0.5
    impl.topk_indices_buffer = torch.tensor([[0, 64, 320]], dtype=torch.int32)

    output, lse = impl.forward_mqa(
        (
            torch.zeros(1, 2, 512, dtype=torch.bfloat16),
            torch.zeros(1, 2, 64, dtype=torch.bfloat16),
        ),
        _cache(),
        metadata,
        SimpleNamespace(_oscar_rotation=torch.eye(512)),
    )

    assert output.dtype == torch.bfloat16
    assert lse is None
    assert captured["selected"].tolist() == [[0, 64, 320]]
    assert captured["query_positions"].tolist() == [320]
    assert captured["block_table"] is metadata.block_table
    assert impl.oscar_read_calls == 1
