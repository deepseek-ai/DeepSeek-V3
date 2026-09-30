"""Real safetensors I/O tests with a CPU dequantizer substituted for Triton.

The CPU tests exercise the actual conversion entry point, not a rewritten
converter. Only the CUDA loader destination and dequantization kernel are
substituted. CUDA coverage below uses the real kernel when hardware exists.

FP8_CONVERTER_SOURCE may point at an unmodified upstream file for comparison.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from safetensors.torch import load_file, save_file


SOURCE = Path(os.environ.get(
    "FP8_CONVERTER_SOURCE",
    Path(__file__).resolve().parents[1] / "fp8_cast_bf16.py",
))
WEIGHT = "model.layers.0.mlp.experts.0.gate_proj.weight"
SCALE = WEIGHT + "_scale_inv"


def load_module():
    spec = importlib.util.spec_from_file_location("fp8_converter_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_dequant(weight, scale, block_size=128):
    rows, cols = weight.shape
    expanded = scale.float().repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return (weight.float() * expanded[:rows, :cols]).to(torch.bfloat16)


@pytest.fixture
def converter(monkeypatch):
    calls = {"loads": [], "dequants": 0, "writes": []}
    kernel = types.ModuleType("kernel")

    def cpu_dequant(weight, scale, block_size=128):
        calls["dequants"] += 1
        return reference_dequant(weight, scale, block_size)

    kernel.weight_dequant = cpu_dequant
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    module = load_module()

    def cpu_load(path, device="cpu"):
        calls["loads"].append(str(path))
        return load_file(path, device="cpu")

    def record_save(tensors, path, **kwargs):
        calls["writes"].append(str(path))
        return save_file(tensors, path, **kwargs)

    monkeypatch.setattr(module, "load_file", cpu_load)
    monkeypatch.setattr(module, "save_file", record_save)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)
    old_dtype = torch.get_default_dtype()
    try:
        yield module, calls
    finally:
        torch.set_default_dtype(old_dtype)


def checkpoint(root, shards, weight_map=None):
    root.mkdir()
    mapping = {}
    for filename, tensors in shards.items():
        save_file(tensors, root / filename)
        mapping.update({name: filename for name in tensors})
    if weight_map is not None:
        mapping = weight_map
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0}, "weight_map": mapping,
    }))
    return mapping


def fp8(shape=(128, 128), dtype=torch.float8_e4m3fn):
    count = 1
    for dim in shape:
        count *= dim
    return ((torch.arange(count, dtype=torch.float32) % 7) - 3).reshape(shape).to(dtype)


def assert_preflight_failure(module, calls, source, output, exception, match):
    before_dtype = torch.get_default_dtype()
    with pytest.raises(exception, match=match):
        module.main(str(source), str(output))
    assert not output.exists(), "preflight must not create partial output"
    assert calls["loads"] == [], "invalid input must fail before CUDA tensor loads"
    assert calls["writes"] == []
    assert calls["dequants"] == 0
    assert torch.get_default_dtype() == before_dtype


@pytest.mark.parametrize("scale_physically_present", [False, True])
def test_missing_scale_index_fails_before_writes(converter, tmp_path, scale_physically_present):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    tensors = {WEIGHT: fp8()}
    if scale_physically_present:
        tensors[SCALE] = torch.ones(1, 1)
    checkpoint(source, {"a.safetensors": tensors}, {WEIGHT: "a.safetensors"})
    assert_preflight_failure(module, calls, source, output, KeyError, "Missing scale_inv")


@pytest.mark.parametrize("indexed_shard", ["b.safetensors", "absent.safetensors"])
def test_wrong_or_missing_indexed_scale_shard(converter, tmp_path, indexed_shard):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, {
        "a.safetensors": {WEIGHT: fp8(), SCALE: torch.ones(1, 1)},
        "b.safetensors": {"model.norm.weight": torch.ones(2)},
    }, {WEIGHT: "a.safetensors", SCALE: indexed_shard,
        "model.norm.weight": "b.safetensors"})
    assert_preflight_failure(module, calls, source, output, KeyError, "Missing scale_inv")


def test_scale_absent_from_existing_indexed_shard(converter, tmp_path):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, {"a.safetensors": {WEIGHT: fp8()}},
               {WEIGHT: "a.safetensors", SCALE: "a.safetensors"})
    assert_preflight_failure(module, calls, source, output, KeyError, "Missing scale_inv")


def test_bad_later_shard_does_not_leave_earlier_outputs(converter, tmp_path):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, {
        "a.safetensors": {"model.norm.weight": torch.ones(4, dtype=torch.bfloat16)},
        "z.safetensors": {WEIGHT: fp8()},
    })
    assert_preflight_failure(module, calls, source, output, KeyError, "Missing scale_inv")


@pytest.mark.parametrize("scale_shape", [(1, 1), (1, 2), (2, 1), (4,), (2, 2, 1)])
def test_malformed_scale_shape_fails_before_device_load(converter, tmp_path, scale_shape):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, {"a.safetensors": {
        WEIGHT: fp8((130, 129)), SCALE: torch.ones(scale_shape),
    }})
    assert_preflight_failure(module, calls, source, output, ValueError, "Invalid scale shape")


def test_nonmatrix_fp8_fails_before_device_load(converter, tmp_path):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, {"a.safetensors": {WEIGHT: fp8((4,)), SCALE: torch.ones(1, 1)}})
    assert_preflight_failure(module, calls, source, output, ValueError, "must be 2-D")


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("shape", [(128, 128), (130, 129)])
@pytest.mark.parametrize("scale_location", ["same", "before", "after"])
def test_valid_conversion_and_cross_shard_scales(converter, tmp_path, dtype, shape, scale_location):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    weight = fp8(shape, dtype)
    grid = tuple((dim + 127) // 128 for dim in shape)
    scale = torch.arange(2, 2 + grid[0] * grid[1], dtype=torch.float32).reshape(grid)
    normal = torch.tensor([1.5, -2.0], dtype=torch.bfloat16)
    shards = {"b.safetensors": {WEIGHT: weight, "model.norm.weight": normal}}
    if scale_location == "same":
        shards["b.safetensors"][SCALE] = scale
    else:
        shards["a.safetensors" if scale_location == "before" else "c.safetensors"] = {SCALE: scale}
    checkpoint(source, shards)
    module.main(str(source), str(output))
    actual = load_file(output / "b.safetensors")
    assert actual[WEIGHT].dtype == torch.bfloat16
    torch.testing.assert_close(actual[WEIGHT], reference_dequant(weight, scale), rtol=0, atol=0)
    torch.testing.assert_close(actual["model.norm.weight"], normal, rtol=0, atol=0)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert SCALE not in index["weight_map"]
    for name, filename in index["weight_map"].items():
        assert name in load_file(output / filename)
    for filename in shards:
        assert SCALE not in load_file(output / filename)
    assert calls["dequants"] == 1


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.bool, torch.float32, torch.bfloat16])
def test_non_fp8_data_is_preserved(converter, tmp_path, dtype):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    weight = torch.ones(4, 4, dtype=dtype)
    checkpoint(source, {"a.safetensors": {"model.aux.weight": weight}})
    module.main(str(source), str(output))
    actual = load_file(output / "a.safetensors")["model.aux.weight"]
    assert actual.dtype == weight.dtype
    torch.testing.assert_close(actual, weight, rtol=0, atol=0)
    assert calls["dequants"] == 0


def test_validation_uses_headers_not_tensor_payloads(converter, tmp_path, monkeypatch):
    module, calls = converter
    source = tmp_path / "source"
    mapping = checkpoint(source, {"a.safetensors": {WEIGHT: fp8(), SCALE: torch.ones(1, 1)}})
    real_safe_open = module.safe_open

    class HeaderOnly:
        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cpu"
            self.handle = real_safe_open(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def keys(self):
            return self.handle.keys()
        def get_slice(self, name):
            return self.handle.get_slice(name)
        def get_tensor(self, name):
            raise AssertionError("preflight must not materialize tensors")

    monkeypatch.setattr(module, "safe_open", HeaderOnly)
    module._validate_fp8_scales(sorted(source.glob("*.safetensors")), mapping)
    assert calls["loads"] == []


def test_existing_output_untouched_on_preflight_failure(converter, tmp_path):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, {"a.safetensors": {WEIGHT: fp8()}})
    output.mkdir()
    marker = output / "model.safetensors.index.json"
    marker.write_text("existing output must not change")
    with pytest.raises(KeyError, match="Missing scale_inv"):
        module.main(str(source), str(output))
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "existing output must not change"
    assert not calls["loads"]
    assert not calls["writes"]


def test_scale_disappearing_after_preflight_does_not_fall_back(converter, tmp_path, monkeypatch):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, {"a.safetensors": {WEIGHT: fp8(), SCALE: torch.ones(1, 1)}})
    real_load = module.load_file

    def lose_scale(*args, **kwargs):
        tensors = real_load(*args, **kwargs)
        tensors.pop(SCALE, None)
        return tensors

    monkeypatch.setattr(module, "load_file", lose_scale)
    with pytest.raises(KeyError):
        module.main(str(source), str(output))
    assert not (output / "model.safetensors.index.json").exists()
    assert not calls["writes"]


def test_kernel_keyerror_is_not_misreported_as_missing_scale(converter, tmp_path, monkeypatch):
    module, calls = converter
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, {"a.safetensors": {WEIGHT: fp8(), SCALE: torch.ones(1, 1)}})

    def fail(*args, **kwargs):
        raise KeyError("simulated kernel failure")

    monkeypatch.setattr(module, "weight_dequant", fail)
    with pytest.raises(KeyError, match="simulated kernel failure"):
        module.main(str(source), str(output))
    assert not (output / "model.safetensors.index.json").exists()
    assert not calls["writes"]


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("CPU_REGRESSION_ONLY") == "1",
    reason="CUDA hardware unavailable or CPU-only regression run")
def test_real_cuda_conversion(tmp_path, monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.syspath_prepend(str(SOURCE.parent))
    module = load_module()
    source, output = tmp_path / "source", tmp_path / "output"
    weight, scale = fp8((130, 129)), torch.tensor([[2., 3.], [4., 5.]])
    checkpoint(source, {"a.safetensors": {WEIGHT: weight}, "b.safetensors": {SCALE: scale}})
    old_dtype = torch.get_default_dtype()
    try:
        module.main(str(source), str(output))
    finally:
        torch.set_default_dtype(old_dtype)
    actual = load_file(output / "a.safetensors")[WEIGHT]
    torch.testing.assert_close(actual, reference_dequant(weight, scale), rtol=0, atol=0)
