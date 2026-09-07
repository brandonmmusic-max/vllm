# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of actual Jovian method selection and the carrier loader ABI.

The runtime launch is replaced by a recorder: these tests prove ownership and
dispatch, not CUDA correctness. No weights or models are downloaded.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import trellismx
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)


@pytest.fixture
def method_inputs(monkeypatch):
    parallel = SimpleNamespace(tp_size=4, ep_size=1, tp_rank=0)
    moe = SimpleNamespace(
        moe_parallel_config=parallel,
        hidden_dim=4096,
        intermediate_size_per_partition=512,
        num_experts=288,
        experts_per_token=8,
        has_bias=False,
        is_lora_enabled=False,
        is_act_and_mul=True,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        swiglu_limit=10.0,
    )
    config = ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
    monkeypatch.setattr(trellismx, "load_overlay", lambda _: object())
    monkeypatch.setattr(
        trellismx,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="glm5_next_text")
            ),
        ),
    )
    return config, moe


def test_opt_in_selects_native_method_before_stock_oracle(method_inputs, monkeypatch):
    config, moe = method_inputs
    monkeypatch.setenv("VLLM_TRELLISMX_CHECKPOINT", "/fixture")
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = moe
    method = config.get_quant_method(layer, "model.layers.3.mlp.experts")
    assert isinstance(method, trellismx.TrellisMXMoEMethod)
    assert not method.is_monolithic
    assert not method.supports_eplb
    assert not method.mk_can_overlap_shared_experts


def test_opt_out_does_not_read_overlay(method_inputs, monkeypatch):
    config, moe = method_inputs
    monkeypatch.delenv("VLLM_TRELLISMX_CHECKPOINT", raising=False)
    assert (
        trellismx.maybe_trellismx_method(
            config, SimpleNamespace(moe_config=moe), "model.layers.3.mlp.experts"
        )
        is None
    )


@pytest.mark.parametrize("field,value", [("tp_size", 2), ("ep_size", 4)])
def test_rejects_unsupported_sharding(method_inputs, field, value):
    config, moe = method_inputs
    setattr(moe.moe_parallel_config, field, value)
    with pytest.raises(ValueError, match="TP4"):
        trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)


def test_inherits_real_modelopt_carrier_weight_loader_shapes(
    method_inputs, monkeypatch
):
    from vllm.model_executor import parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 4)
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    layer = torch.nn.Module()
    # Exercise the real loader with tiny tensors instead of 288 experts.
    method.create_weights(layer, 2, 32, 16, torch.bfloat16)
    assert layer.w13_weight.shape == (2, 32, 16)
    assert layer.w2_weight.shape == (2, 32, 8)
    assert layer.w13_weight_scale.dtype == torch.float8_e4m3fn
    assert method.uses_weight_scale_2_pattern()


def test_dispatch_passes_only_routed_input_and_routes(method_inputs):
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    x = torch.empty((1, 4096), dtype=torch.bfloat16)
    weights, ids, shared = object(), object(), object()
    seen = []

    def record(*args):
        seen.append(args)
        return "routed result"

    method.runtime = record
    result = method.apply(None, x, weights, ids, shared, shared)
    assert seen == [(x, weights, ids)]
    assert result == "routed result"
    with pytest.raises(RuntimeError, match="routing belongs"):
        method.apply_monolithic()


def test_empty_dispatch_does_not_launch(method_inputs):
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    method.runtime = lambda *args: pytest.fail("empty batch launched")
    result = method.apply(None, torch.empty((0, 4096)), None, None, None, None)
    assert result.shape == (0, 4096)


def test_missing_runtime_cannot_serve_carrier(method_inputs):
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    with pytest.raises(RuntimeError, match="not loaded"):
        method.apply(None, None, None, None, None, None)


def test_native_runtime_operator_namespace_matches_calls():
    from trellismx_runtime import p8_native_kernel

    assert p8_native_kernel.P8NativeTPMoE is not None
    assert hasattr(torch.ops.trellismx_b12x, "dense_gemm_launch")
    assert hasattr(torch.ops.trellismx_b12x, "tp_moe_dynamic_launch")
    assert not hasattr(torch.ops.trellismx_trellismx_b12x, "dense_gemm_launch")


def test_warmup_covers_decode_verification_and_prefill_without_dedup(method_inputs):
    config, moe = method_inputs
    calls = []

    class Recorder:
        device = "cpu"

        def __call__(self, x, weights, ids):
            calls.append((tuple(x.shape), tuple(weights.shape), tuple(ids.shape)))

    keys = []
    for index in (3, 4):
        method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", index)
        method.runtime = Recorder()
        unit = method.get_b12x_warmup_unit(None, (1, 4, 17), torch.bfloat16)
        keys.append(unit.key)
        unit.compile()
    assert keys[0] != keys[1]
    assert [call[0][0] for call in calls] == [1, 4, 17, 1, 4, 17]
    assert all(call[1][1] == call[2][1] == 8 for call in calls)


def test_mtp_alias_is_opt_in_and_does_not_select_p8(method_inputs, monkeypatch):
    config, moe = method_inputs
    prefix = "model.layers.45.mtp_block.mlp.experts"
    canonical = "model.language_model.layers.45.mlp.experts"
    monkeypatch.delenv("VLLM_TRELLISMX_CHECKPOINT", raising=False)
    assert (
        canonical
        not in ModelOptMixedPrecisionConfig._quantized_layer_prefix_candidates(prefix)
    )
    monkeypatch.setenv("VLLM_TRELLISMX_CHECKPOINT", "/fixture")
    assert canonical in ModelOptMixedPrecisionConfig._quantized_layer_prefix_candidates(
        prefix
    )
    assert (
        trellismx.maybe_trellismx_method(
            config, SimpleNamespace(moe_config=moe), prefix
        )
        is None
    )
