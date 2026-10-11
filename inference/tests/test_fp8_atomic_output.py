"""Regression tests for safe FP8 conversion output publication.

The dequantizer is replaced with a deterministic CPU implementation. These
tests cover file/index handling and failure safety, not the production CUDA
kernel's numerical correctness.
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


SOURCE = Path(__file__).resolve().parents[1] / "fp8_cast_bf16.py"


@pytest.fixture
def converter(monkeypatch):
    kernel = types.ModuleType("kernel")

    def cpu_dequant(weight, scale):
        return (weight.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)

    kernel.weight_dequant = cpu_dequant
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    spec = importlib.util.spec_from_file_location("fp8_converter_atomic_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    real_load_file = module.load_file
    monkeypatch.setattr(
        module,
        "load_file",
        lambda path, device="cpu": real_load_file(path, device="cpu"),
    )
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)
    return module


def make_fp8_input(root):
    root.mkdir()
    first_weight = torch.tensor([1.0, 2.0, -3.0, 4.0], dtype=torch.float8_e4m3fn)
    second_weight = torch.tensor([5.0, -6.0, 7.0, 8.0], dtype=torch.float8_e4m3fn)

    # Intentionally put each weight's scale in the other shard to cover
    # cross-shard scale lookup.
    save_file(
        {
            "model.first.weight": first_weight,
            "model.second.weight_scale_inv": torch.ones(1, dtype=torch.float32),
        },
        root / "a.safetensors",
    )
    save_file(
        {
            "model.first.weight_scale_inv": torch.ones(1, dtype=torch.float32),
            "model.second.weight": second_weight,
        },
        root / "b.safetensors",
    )
    index = {
        "metadata": {"total_size": 999, "audit_marker": "preserve-me"},
        "weight_map": {
            "model.first.weight": "a.safetensors",
            "model.first.weight_scale_inv": "b.safetensors",
            "model.second.weight_scale_inv": "a.safetensors",
            "model.second.weight": "b.safetensors",
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def temporary_outputs(parent, output_name):
    return list(parent.glob(f".{output_name}.staging-*"))


def test_later_shard_write_failure_cleans_staging_and_restores_dtype(
    converter, monkeypatch, tmp_path
):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    make_fp8_input(source)

    old_dtype = torch.get_default_dtype()
    real_save_file = converter.save_file
    writes = 0

    def fail_on_second_shard(tensors, path, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated later-shard write failure")
        return real_save_file(tensors, path, **kwargs)

    monkeypatch.setattr(converter, "save_file", fail_on_second_shard)
    with pytest.raises(OSError, match="simulated later-shard"):
        converter.main(str(source), str(output))

    assert not output.exists()
    assert temporary_outputs(tmp_path, output.name) == []
    assert torch.get_default_dtype() == old_dtype


def test_existing_destination_is_rejected_and_preserved(converter, tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    make_fp8_input(source)
    output.mkdir()
    (output / "sentinel.txt").write_text("previous output", encoding="utf-8")

    with pytest.raises(FileExistsError, match="never overwritten"):
        converter.main(str(source), str(output))

    assert sorted(path.name for path in output.iterdir()) == ["sentinel.txt"]
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "previous output"
    assert temporary_outputs(tmp_path, output.name) == []


def test_success_publishes_complete_output_and_updates_metadata(converter, tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    make_fp8_input(source)

    converter.main(str(source), str(output))

    assert sorted(path.name for path in output.iterdir()) == [
        "a.safetensors",
        "b.safetensors",
        "model.safetensors.index.json",
    ]
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert set(index["weight_map"]) == {
        "model.first.weight",
        "model.second.weight",
    }
    assert index["metadata"]["audit_marker"] == "preserve-me"
    assert index["metadata"]["total_size"] == 16

    converted_a = load_file(output / "a.safetensors", device="cpu")
    converted_b = load_file(output / "b.safetensors", device="cpu")
    assert converted_a["model.first.weight"].dtype == torch.bfloat16
    assert converted_b["model.second.weight"].dtype == torch.bfloat16
    assert "model.second.weight_scale_inv" not in converted_a
    assert "model.first.weight_scale_inv" not in converted_b
    assert temporary_outputs(tmp_path, output.name) == []


def test_non_fp8_one_byte_tensor_is_not_sent_to_dequantizer(converter, tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    source.mkdir()
    save_file(
        {"model.quantized_int8": torch.tensor([1, -2, 3], dtype=torch.int8)},
        source / "weights.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.quantized_int8": "weights.safetensors"}}),
        encoding="utf-8",
    )

    converter.main(str(source), str(output))

    weights = load_file(output / "weights.safetensors", device="cpu")
    assert weights["model.quantized_int8"].dtype == torch.int8
    assert weights["model.quantized_int8"].tolist() == [1, -2, 3]


def test_publish_rename_failure_cleans_stage_and_leaves_no_destination(
    converter, monkeypatch, tmp_path
):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    make_fp8_input(source)

    old_dtype = torch.get_default_dtype()
    real_replace = os.replace

    def fail_publish_once(src, dst):
        if ".converted.staging-" in str(src) and str(dst) == str(output):
            raise OSError("simulated publication rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(converter.os, "replace", fail_publish_once)
    with pytest.raises(OSError, match="simulated publication rename failure"):
        converter.main(str(source), str(output))

    assert not output.exists()
    assert temporary_outputs(tmp_path, output.name) == []
    assert torch.get_default_dtype() == old_dtype


def test_unindexed_tensor_in_output_shard_prevents_publication(converter, tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    source.mkdir()
    save_file(
        {
            "model.quantized_int8": torch.tensor([1, -2, 3], dtype=torch.int8),
            "model.orphan": torch.tensor([99], dtype=torch.int8),
        },
        source / "weights.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.quantized_int8": "weights.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unindexed"):
        converter.main(str(source), str(output))

    assert not output.exists()
    assert temporary_outputs(tmp_path, output.name) == []


def test_missing_scale_fails_without_publishing_output(converter, tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    source.mkdir()
    save_file(
        {"model.weight": torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)},
        source / "weights.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "weights.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="scale_inv"):
        converter.main(str(source), str(output))

    assert not output.exists()
    assert temporary_outputs(tmp_path, output.name) == []


def test_input_output_overlap_is_rejected(converter, tmp_path):
    source = tmp_path / "input"
    make_fp8_input(source)
    output_inside_source = source / "converted"

    with pytest.raises(ValueError, match="must not contain or overlap"):
        converter.main(str(source), str(output_inside_source))

    assert not output_inside_source.exists()
