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
                    enable_dbo=False,
                )
            },
            "decode context",
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
            enable_dbo=False,
        ),
    )
    for name, value in override.items():
        setattr(config, name, value)

    with pytest.raises(ValueError, match=reason):
        VllmConfig._validate_oscar_mla_runtime(config)
