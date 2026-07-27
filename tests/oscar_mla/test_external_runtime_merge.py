from types import SimpleNamespace

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_prefill_shape_bucket_rounding_and_guards() -> None:
    runner = SimpleNamespace(
        prefill_shape_bucket_enabled=True,
        prefill_shape_bucket_multiple=128,
        prefill_shape_bucket_max=384,
        max_num_tokens=512,
        uniform_decode_query_len=1,
    )

    pad = GPUModelRunner._pad_for_prefill_shape_bucket
    assert pad(runner, 129, 512, False) == 256
    assert pad(runner, 300, 512, False) == 384
    assert pad(runner, 384, 512, False) == 384
    assert pad(runner, 129, 1, False) == 129
    assert pad(runner, 129, 512, True) == 129

    runner.prefill_shape_bucket_enabled = False
    assert pad(runner, 129, 512, False) == 129


def test_shape_metadata_replace_preserves_oscar_ownership() -> None:
    oscar_ownership = object()
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        oscar_mla=oscar_ownership,
    )

    padded = metadata.replace(max_query_len=4, max_seq_len=4)

    assert padded.oscar_mla is oscar_ownership
    assert padded.max_query_len == 4
    assert padded.max_seq_len == 4
    assert metadata.max_query_len == 2
