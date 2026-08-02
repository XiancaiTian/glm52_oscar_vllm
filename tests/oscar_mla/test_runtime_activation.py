# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
)
from vllm.v1.attention.ops import triton_oscar_mla_decode as oscar_decode
from vllm.v1.attention.ops import triton_oscar_mla_store as oscar_store
from vllm.v1.attention.ops.triton_oscar_mla_decode import (
    oscar_mla_sparse_decode,
    oscar_mla_sparse_prefill,
)
from vllm.v1.kv_cache_interface import OscarMLAAttentionSpec


def test_oscar_cache_dtype_is_explicit_and_fail_closed() -> None:
    config = CacheConfig(
        cache_dtype="oscar_mla_int2",
        enable_prefix_caching=False,
    )

    assert config.cache_dtype == "oscar_mla_int2"
    assert STR_DTYPE_TO_TORCH_DTYPE[config.cache_dtype] is torch.uint8
    assert TritonMLASparseBackend.supports_kv_cache_dtype(config.cache_dtype)

    with pytest.raises(ValueError, match="does not support vLLM prefix caching"):
        CacheConfig(cache_dtype="oscar_mla_int2", enable_prefix_caching=True)


def test_mla_layer_builds_oscar_three_pool_spec() -> None:
    layer = SimpleNamespace(
        kv_cache_dtype="oscar_mla_int2",
        use_sparse=True,
        attn_backend=TritonMLASparseBackend,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        head_size=576,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=False,
        ),
    )

    spec = MLAAttention.get_kv_cache_spec(layer, vllm_config)

    assert isinstance(spec, OscarMLAAttentionSpec)
    assert spec.latent_rank == 512
    assert spec.rope_head_size == 64
    assert spec.history_slot_size == 160
    assert spec.prefix_tokens == 64
    assert spec.recent_tokens == 256


@pytest.mark.parametrize(
    "attention",
    [oscar_mla_sparse_decode, oscar_mla_sparse_prefill],
)
def test_oscar_sparse_attention_accepts_keyword_only_inverse_rotation(
    attention,
) -> None:
    parameter = inspect.signature(attention).parameters["inverse_rotation"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_oscar_inverse_rotation_fusion_contract() -> None:
    assert hasattr(oscar_store, "oscar_mla_rotate_add")
    rotate_add = oscar_store.oscar_mla_rotate_add
    parameters = inspect.signature(rotate_add).parameters

    assert tuple(parameters) == ("latent", "rotation", "addend", "output")
    assert parameters["output"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["output"].default is None

    source = inspect.getsource(oscar_decode._oscar_mla_sparse_attention)
    assert "oscar_mla_rotate_add(" in source
    assert "_add_outputs_kernel" not in source


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"model_config": SimpleNamespace(enforce_eager=False)}, "non-eager"),
        (
            {
                "compilation_config": SimpleNamespace(
                    cudagraph_mode=CUDAGraphMode.PIECEWISE
                )
            },
            "CUDA graph",
        ),
        ({"speculative_config": SimpleNamespace()}, "speculative"),
        (
            {
                "parallel_config": SimpleNamespace(
                    decode_context_parallel_size=2,
                    prefill_context_parallel_size=1,
                    enable_dbo=False,
                )
            },
            "decode context",
        ),
        (
            {
                "parallel_config": SimpleNamespace(
                    decode_context_parallel_size=1,
                    prefill_context_parallel_size=2,
                    enable_dbo=False,
                )
            },
            "prefill context",
        ),
        ({"kv_transfer_config": SimpleNamespace()}, "KV transfer"),
        (
            {"scheduler_config": SimpleNamespace(async_scheduling=True)},
            "asynchronous scheduling",
        ),
        (
            {"cache_config": SimpleNamespace(kv_offloading_size=8)},
            "KV offloading",
        ),
    ],
)
def test_oscar_runtime_rejects_unimplemented_modes(
    override: dict[str, object],
    reason: str,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(kv_offloading_size=None),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    for name, value in override.items():
        setattr(config, name, value)

    with pytest.raises(ValueError, match=reason):
        VllmConfig._validate_oscar_mla_runtime(config)
