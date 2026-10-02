"""CPU unit tests for scale-format propagation; Triton kernels are not executed."""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def model(monkeypatch):
    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("A GPU kernel was called outside the CPU unit oracle")

    kernel = types.ModuleType("kernel")
    kernel.act_quant = kernel.weight_dequant = kernel.fp8_gemm = unexpected_kernel
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "deepseek_native_model", root / "inference" / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def activation_quantization_oracle(x, block_size=128, scale_fmt=None):
    blocks = x.float().reshape(*x.shape[:-1], -1, block_size)
    scale = blocks.abs().amax(dim=-1).clamp_min(1e-4) / 448.0
    if scale_fmt == "ue8m0":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape_as(x)
    return quantized, scale


def weight_dequantization_oracle(weight, scale):
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * expanded[: weight.shape[0], : weight.shape[1]]


@pytest.mark.parametrize(
    "layer_name", ["Linear", "ColumnParallelLinear", "RowParallelLinear"]
)
@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("world_size", [1, 2])
def test_native_parallel_forward_applies_configured_activation_scale(
    model, monkeypatch, layer_name, scale_fmt, bias, world_size
):
    torch.manual_seed(73)
    model.world_size = world_size
    model.Linear.scale_fmt = scale_fmt
    model.gemm_impl = "fp8"
    layer = getattr(model, layer_name)(256, 256, bias=bias, dtype=torch.float8_e4m3fn)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape).to(torch.float8_e4m3fn))
        layer.weight.scale.fill_(0.0625)
        if bias:
            layer.bias.copy_(torch.randn_like(layer.bias))
    x = torch.randn(2, layer.weight.shape[1]) * 3.1
    received_formats = []

    def act_quant(x, block_size, actual_format):
        received_formats.append(actual_format)
        return activation_quantization_oracle(x, block_size, actual_format)

    def gemm(quantized, scale, weight, weight_scale):
        dequantized = quantized.float() * scale.repeat_interleave(128, dim=-1)
        return F.linear(dequantized, weight_dequantization_oracle(weight, weight_scale))

    reductions = []

    def all_reduce(tensor):
        reductions.append(tensor.shape)
        # Identical test inputs and weights on each mock rank make a sum equal to world_size times local output.
        tensor.mul_(world_size)

    monkeypatch.setattr(model, "act_quant", act_quant)
    monkeypatch.setattr(model, "fp8_gemm", gemm)
    monkeypatch.setattr(model.dist, "all_reduce", all_reduce)
    actual = layer(x)

    quantized, scale = activation_quantization_oracle(x, model.block_size, scale_fmt)
    dequantized = quantized.float() * scale.repeat_interleave(128, dim=-1)
    expected = F.linear(
        dequantized, weight_dequantization_oracle(layer.weight, layer.weight.scale)
    )
    if layer_name == "RowParallelLinear":
        expected = expected * world_size
    if bias:
        expected = expected + layer.bias
    torch.testing.assert_close(actual, expected)
    assert received_formats == [scale_fmt]
    assert len(reductions) == int(layer_name == "RowParallelLinear" and world_size > 1)


@pytest.mark.parametrize(
    "layer_name", ["Linear", "ColumnParallelLinear", "RowParallelLinear"]
)
@pytest.mark.parametrize("bias", [False, True])
def test_unquantized_native_forward_keeps_bias_and_values(model, layer_name, bias):
    torch.manual_seed(91)
    model.Linear.scale_fmt = "ue8m0"
    layer = getattr(model, layer_name)(8, 8, bias=bias, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.normal_()
        if bias:
            layer.bias.normal_()
    x = torch.randn(3, 8)
    torch.testing.assert_close(layer(x), F.linear(x, layer.weight, layer.bias))
