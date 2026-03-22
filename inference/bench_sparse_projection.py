from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import torch

from dim_gate import HiddenGateOutput, HiddenGateStats
from mla_native import native_sparse_project_hidden
from model import ColumnParallelLinear, Linear
from train_gate_compare import default_dtype_for, select_device


def maybe_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def build_gate_output(
    batch_size: int,
    seq_len: int,
    dim: int,
    active_count: int,
    device: torch.device,
) -> HiddenGateOutput:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)
    active_count = max(1, min(dim, active_count))
    flat_tokens = batch_size * seq_len
    rows = []
    for _ in range(flat_tokens):
        rows.append(torch.randperm(dim, generator=gen)[:active_count])
    active_indices = torch.stack(rows, dim=0).view(batch_size, seq_len, active_count).to(device=device, dtype=torch.long)
    selection_mask = torch.zeros(batch_size, seq_len, dim, dtype=torch.bool, device=device)
    selection_mask.scatter_(2, active_indices, True)
    active_counts = torch.full((batch_size, seq_len), active_count, dtype=torch.long, device=device)
    native_active_indices = active_indices.to(dtype=torch.int32).contiguous() if device.type == "mps" else None
    native_active_counts = active_counts.to(dtype=torch.int32).contiguous() if device.type == "mps" else None
    gate_input = torch.zeros(batch_size, seq_len, dim, dtype=torch.float32, device=device)
    return HiddenGateOutput(
        selection_mask=selection_mask,
        stats=HiddenGateStats(),
        gate_input=gate_input,
        active_indices=active_indices,
        active_counts=active_counts,
        native_active_indices=native_active_indices,
        native_active_counts=native_active_counts,
    )


def benchmark_case(
    name: str,
    proj: torch.nn.Module,
    x: torch.Tensor,
    gate_output: HiddenGateOutput,
    warmup: int,
    iters: int,
    device: torch.device,
) -> Dict[str, float]:
    masked_x = x * gate_output.selection_mask.to(dtype=x.dtype)
    with torch.inference_mode():
        dense_ref = proj(masked_x)
        native_ref = native_sparse_project_hidden(
            x,
            gate_output.selection_mask,
            gate_output.native_active_indices if gate_output.native_active_indices is not None else gate_output.active_indices.contiguous(),
            gate_output.native_active_counts if gate_output.native_active_counts is not None else gate_output.active_counts.contiguous(),
            proj.weight,
            getattr(proj, "bias", None),
        )
    max_diff = float((dense_ref - native_ref).abs().max().item())

    with torch.inference_mode():
        for _ in range(warmup):
            _ = proj(masked_x)
            _ = native_sparse_project_hidden(
                x,
                gate_output.selection_mask,
                gate_output.native_active_indices if gate_output.native_active_indices is not None else gate_output.active_indices.contiguous(),
                gate_output.native_active_counts if gate_output.native_active_counts is not None else gate_output.active_counts.contiguous(),
                proj.weight,
                getattr(proj, "bias", None),
            )
        maybe_sync(device)

        start = time.perf_counter()
        for _ in range(iters):
            _ = proj(masked_x)
        maybe_sync(device)
        dense_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(iters):
            _ = native_sparse_project_hidden(
                x,
                gate_output.selection_mask,
                gate_output.native_active_indices if gate_output.native_active_indices is not None else gate_output.active_indices.contiguous(),
                gate_output.native_active_counts if gate_output.native_active_counts is not None else gate_output.active_counts.contiguous(),
                proj.weight,
                getattr(proj, "bias", None),
            )
        maybe_sync(device)
        native_elapsed = time.perf_counter() - start

    return {
        "dense_ms": dense_elapsed * 1000.0 / iters,
        "native_ms": native_elapsed * 1000.0 / iters,
        "speedup": dense_elapsed / native_elapsed if native_elapsed > 0 else 0.0,
        "max_diff": max_diff,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=224)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--active-count", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    torch.set_default_dtype(default_dtype_for(device, "fp32"))
    x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=torch.float32)
    gate_output = build_gate_output(args.batch_size, args.seq_len, args.dim, args.active_count, device)

    cases = {
        "wq": ColumnParallelLinear(args.dim, 96, dtype=torch.float32).to(device),
        "wq_a": Linear(args.dim, 16, dtype=torch.float32).to(device),
        "wkv_a": Linear(args.dim, 40, dtype=torch.float32).to(device),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, proj in cases.items():
        proj.eval()
        results[name] = benchmark_case(name, proj, x, gate_output, args.warmup, args.iters, device)
        print(
            f"{name} dense_ms={results[name]['dense_ms']:.3f} native_ms={results[name]['native_ms']:.3f} "
            f"speedup={results[name]['speedup']:.3f} max_diff={results[name]['max_diff']:.3e}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "config": {
                        "device": str(device),
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "dim": args.dim,
                        "active_count": args.active_count,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    },
                    "results": results,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
