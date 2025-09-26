import os
import json
import tempfile
from typing import Tuple

import torch
from safetensors.torch import save_file, load_file

# Import the conversion API we expose for programmatic use
from inference.fp8_cast_bf16 import convert_fp8_to_bf16


def _make_block_scale(shape_blocks: Tuple[int, int], value: float, device: str) -> torch.Tensor:
    """
    Create a per-block scale tensor of shape (M_blocks, N_blocks) filled with a constant.
    """
    return torch.full(shape_blocks, value, dtype=torch.float32, device=device).contiguous()


def test_convert_fp8_to_bf16_cpu_roundtrip_small_matrix():
    """
    Validate CPU fallback by constructing a tiny FP8 weight with known block scales,
    converting to BF16, and checking the recovered values.
    """
    if not hasattr(torch, "float8_e4m3fn"):
        # Skip if PyTorch build lacks float8 support
        return

    device = "cpu"
    block_size = 2
    M, N = 4, 4
    # Choose a uniform block scale that is easy to reason about
    scale_value = 0.5  # multiplicative factor used during dequant
    # Construct the target dequantized weights (what we want to recover)
    y_true = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [2.0, 4.0, 6.0, 8.0],
         [1.5, 2.5, 3.5, 4.5]],
        dtype=torch.float32,
        device=device,
    )
    # Create the per-block scale tensor: (M // block_size, N // block_size)
    s = _make_block_scale((M // block_size, N // block_size), scale_value, device)
    # Expand s to full resolution for constructing FP8 quantized weights
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    # Build FP8 weights such that dequant (x * s_full) recovers y_true
    x_fp32 = (y_true / scale_value).contiguous()
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    with tempfile.TemporaryDirectory() as tmp:
        fp8_dir = os.path.join(tmp, "fp8")
        bf16_dir = os.path.join(tmp, "bf16")
        os.makedirs(fp8_dir, exist_ok=True)
        os.makedirs(bf16_dir, exist_ok=True)

        # Create minimal safetensors shard and index
        shard = {"layer.weight": x_fp8, "layer.weight_scale_inv": s}
        shard_name = "model-00001-of-00001.safetensors"
        save_file(shard, os.path.join(fp8_dir, shard_name))

        index = {"metadata": {}, "weight_map": {"layer.weight": shard_name, "layer.weight_scale_inv": shard_name}}
        with open(os.path.join(fp8_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        # Run conversion using CPU path and a small block size
        convert_fp8_to_bf16(fp8_dir, bf16_dir, device="cpu", block_size=block_size)

        # Load converted weights and verify they match the expected y_true (within tolerance)
        out_shard = load_file(os.path.join(bf16_dir, shard_name), device=device)
        y = out_shard["layer.weight"].to(torch.float32)
        assert torch.allclose(y, y_true, atol=1e-2, rtol=1e-2)


