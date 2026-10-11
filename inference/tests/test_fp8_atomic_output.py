"""Atomic publication regression tests using real safetensors files."""
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
    kernel.weight_dequant = lambda weight, scale: weight.to(torch.bfloat16)
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    spec = importlib.util.spec_from_file_location("fp8_converter_atomic_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original_load = module.load_file
    monkeypatch.setattr(
        module, "load_file", lambda path, device="cpu": original_load(path, device="cpu")
    )
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)
    return module


def make_input(root):
    root.mkdir()
    first = {"model.first.weight": torch.arange(4, dtype=torch.float32)}
    later = {"model.later.weight": torch.arange(4, dtype=torch.float32) + 10}
    save_file(first, root / "a.safetensors")
    save_file(later, root / "b.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.first.weight": "a.safetensors",
            "model.later.weight": "b.safetensors",
        }
    }))


def temporary_outputs(parent, output_name):
    return list(parent.glob(f".{output_name}.staging-*")) + list(parent.glob(f".{output_name}.backup-*"))


@pytest.mark.parametrize("preexisting", [False, True])
def test_later_shard_write_failure_preserves_destination_and_cleans_stage(
    converter, monkeypatch, tmp_path, preexisting
):
    module = converter
    source = tmp_path / "input"
    output = tmp_path / "converted"
    make_input(source)
    if preexisting:
        output.mkdir()
        (output / "sentinel").write_text("previous output")

    real_save = module.save_file
    writes = 0

    def fail_on_second_shard(tensors, path, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated later-shard write failure")
        return real_save(tensors, path, **kwargs)

    monkeypatch.setattr(module, "save_file", fail_on_second_shard)
    with pytest.raises(OSError, match="simulated later-shard"):
        module.main(str(source), str(output))

    assert temporary_outputs(tmp_path, output.name) == []
    if preexisting:
        assert sorted(p.name for p in output.iterdir()) == ["sentinel"]
        assert (output / "sentinel").read_text() == "previous output"
    else:
        assert not output.exists()


def test_success_replaces_old_directory_and_removes_stale_files(converter, tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    make_input(source)
    output.mkdir()
    (output / "stale.safetensors").write_text("stale")

    converter.main(str(source), str(output))

    assert sorted(p.name for p in output.iterdir()) == [
        "a.safetensors", "b.safetensors", "model.safetensors.index.json",
    ]
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"model.first.weight", "model.later.weight"}
    assert temporary_outputs(tmp_path, output.name) == []


def test_publication_rename_failure_rolls_back_existing_directory(converter, monkeypatch, tmp_path):
    module = converter
    source = tmp_path / "input"
    destination = tmp_path / "output"
    make_input(source)
    destination.mkdir()
    (destination / "sentinel").write_text("previous")
    real_replace = os.replace

    def fail_publish_once(src, dst):
        if ".output.staging-" in str(src) and str(dst) == str(destination):
            raise OSError("simulated publish rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(module.os, "replace", fail_publish_once)
    with pytest.raises(OSError, match="simulated publish rename failure"):
        module.main(str(source), str(destination))

    assert (destination / "sentinel").read_text() == "previous"
    assert not (destination / "a.safetensors").exists()
    assert temporary_outputs(tmp_path, destination.name) == []
