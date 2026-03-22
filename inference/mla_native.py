from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch


MAX_NATIVE_ACTIVE = 512

_EXTENSION = None
_LOAD_ERROR: Optional[Exception] = None


def _load_extension():
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None or _LOAD_ERROR is not None:
        return _EXTENSION

    try:
        from torch.utils.cpp_extension import load

        base_dir = Path(__file__).resolve().parent
        source = base_dir / "native" / "sparse_projection_mps.mm"
        build_dir = base_dir / "native" / "build_sparse_projection"
        build_dir.mkdir(parents=True, exist_ok=True)
        venv_bin = str(Path(sys.prefix) / "bin")
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        _EXTENSION = load(
            name="sparse_projection_mps_ext",
            sources=[str(source)],
            extra_cflags=["-std=c++17", "-fobjc-arc"],
            extra_ldflags=[
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-framework",
                "MetalPerformanceShaders",
                "-framework",
                "MetalPerformanceShadersGraph",
            ],
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - toolchain-specific
        _LOAD_ERROR = exc
        _EXTENSION = None
    return _EXTENSION


def mps_sparse_projection_available() -> bool:
    if not torch.backends.mps.is_available():
        return False
    return _load_extension() is not None


def mps_sparse_projection_error() -> Optional[Exception]:
    _load_extension()
    return _LOAD_ERROR


def run_mps_sparse_projection(
    x: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> Optional[torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.sparse_projection_mps(x, active_indices, active_counts, weight, bias)


def run_mps_sparse_projection_pair(
    x: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    weight_a: torch.Tensor,
    bias_a: torch.Tensor,
    weight_b: torch.Tensor,
    bias_b: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.sparse_projection_pair_mps(
        x, active_indices, active_counts, weight_a, bias_a, weight_b, bias_b
    )


def _ensure_int32_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.int32 and tensor.is_contiguous():
        return tensor
    return tensor.to(dtype=torch.int32).contiguous()


class SparseProjectionMPSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        selection_mask: torch.Tensor,
        active_indices: torch.Tensor,
        active_counts: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        if selection_mask.shape != x.shape:
            raise ValueError(
                f"selection mask shape {tuple(selection_mask.shape)} does not match input {tuple(x.shape)}"
            )
        if active_indices.shape[:2] != (bsz, seqlen):
            raise ValueError(
                f"active index metadata shape {tuple(active_indices.shape[:2])} does not match input {(bsz, seqlen)}"
            )
        if active_counts.shape != (bsz, seqlen):
            raise ValueError(
                f"active count shape {tuple(active_counts.shape)} does not match input {(bsz, seqlen)}"
            )

        x_flat = x.reshape(-1, dim).contiguous()
        idx_flat = _ensure_int32_contiguous(active_indices.reshape(-1, active_indices.size(-1)))
        count_flat = _ensure_int32_contiguous(active_counts.reshape(-1))
        weight_c = weight.contiguous()
        bias_c = bias.contiguous()

        output = run_mps_sparse_projection(x_flat, idx_flat, count_flat, weight_c, bias_c)
        if output is None:
            raise RuntimeError("native sparse projection MPS extension is unavailable")

        # v1 keeps forward native and backward exact-but-dense in PyTorch. That keeps
        # training correct without having to ship a second custom kernel immediately.
        needs_backward_state = ctx.needs_input_grad[0] or ctx.needs_input_grad[4] or ctx.needs_input_grad[5]
        ctx.has_backward_state = needs_backward_state
        if needs_backward_state:
            masked_x = x * selection_mask.to(dtype=x.dtype)
            ctx.save_for_backward(masked_x, selection_mask, weight_c, bias_c)
        return output.view(bsz, seqlen, weight.size(0))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not getattr(ctx, "has_backward_state", False):
            raise RuntimeError("sparse projection backward state was not saved")
        masked_x, selection_mask, weight, bias = ctx.saved_tensors
        grad_out_flat = grad_output.reshape(-1, grad_output.size(-1))
        masked_x_flat = masked_x.reshape(-1, masked_x.size(-1))

        grad_input = grad_out_flat.matmul(weight).view_as(masked_x)
        grad_input = grad_input * selection_mask.to(dtype=grad_input.dtype)
        grad_weight = grad_out_flat.t().matmul(masked_x_flat)
        grad_bias = grad_out_flat.sum(dim=0) if bias.numel() > 0 else None
        return grad_input, None, None, None, grad_weight, grad_bias


class SparseProjectionPairMPSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        selection_mask: torch.Tensor,
        active_indices: torch.Tensor,
        active_counts: torch.Tensor,
        weight_a: torch.Tensor,
        bias_a: torch.Tensor,
        weight_b: torch.Tensor,
        bias_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, dim = x.shape
        if selection_mask.shape != x.shape:
            raise ValueError(
                f"selection mask shape {tuple(selection_mask.shape)} does not match input {tuple(x.shape)}"
            )
        if active_indices.shape[:2] != (bsz, seqlen):
            raise ValueError(
                f"active index metadata shape {tuple(active_indices.shape[:2])} does not match input {(bsz, seqlen)}"
            )
        if active_counts.shape != (bsz, seqlen):
            raise ValueError(
                f"active count shape {tuple(active_counts.shape)} does not match input {(bsz, seqlen)}"
            )

        x_flat = x.reshape(-1, dim).contiguous()
        idx_flat = _ensure_int32_contiguous(active_indices.reshape(-1, active_indices.size(-1)))
        count_flat = _ensure_int32_contiguous(active_counts.reshape(-1))
        weight_a_c = weight_a.contiguous()
        bias_a_c = bias_a.contiguous()
        weight_b_c = weight_b.contiguous()
        bias_b_c = bias_b.contiguous()

        outputs = run_mps_sparse_projection_pair(
            x_flat, idx_flat, count_flat, weight_a_c, bias_a_c, weight_b_c, bias_b_c
        )
        if outputs is None:
            raise RuntimeError("native sparse projection pair MPS extension is unavailable")
        output_a, output_b = outputs

        # The fused pair path shares one sparse forward launch but still uses analytic
        # dense gradients so it remains easy to validate against the reference path.
        needs_backward_state = any(
            ctx.needs_input_grad[index] for index in (0, 4, 5, 6, 7)
        )
        ctx.has_backward_state = needs_backward_state
        if needs_backward_state:
            masked_x = x * selection_mask.to(dtype=x.dtype)
            ctx.save_for_backward(
                masked_x, selection_mask, weight_a_c, bias_a_c, weight_b_c, bias_b_c
            )
        return (
            output_a.view(bsz, seqlen, weight_a.size(0)),
            output_b.view(bsz, seqlen, weight_b.size(0)),
        )

    @staticmethod
    def backward(ctx, grad_output_a: torch.Tensor, grad_output_b: torch.Tensor):
        if not getattr(ctx, "has_backward_state", False):
            raise RuntimeError("sparse projection pair backward state was not saved")
        masked_x, selection_mask, weight_a, bias_a, weight_b, bias_b = ctx.saved_tensors
        grad_a_flat = grad_output_a.reshape(-1, grad_output_a.size(-1))
        grad_b_flat = grad_output_b.reshape(-1, grad_output_b.size(-1))
        masked_x_flat = masked_x.reshape(-1, masked_x.size(-1))

        grad_input = grad_a_flat.matmul(weight_a) + grad_b_flat.matmul(weight_b)
        grad_input = grad_input.view_as(masked_x)
        grad_input = grad_input * selection_mask.to(dtype=grad_input.dtype)
        grad_weight_a = grad_a_flat.t().matmul(masked_x_flat)
        grad_bias_a = grad_a_flat.sum(dim=0) if bias_a.numel() > 0 else None
        grad_weight_b = grad_b_flat.t().matmul(masked_x_flat)
        grad_bias_b = grad_b_flat.sum(dim=0) if bias_b.numel() > 0 else None
        return grad_input, None, None, None, grad_weight_a, grad_bias_a, grad_weight_b, grad_bias_b


def native_sparse_project_hidden(
    x: torch.Tensor,
    selection_mask: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    requires_grad = (
        torch.is_grad_enabled()
        and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad))
    )
    if not requires_grad:
        x_flat = x.reshape(-1, x.size(-1)).contiguous()
        idx_flat = _ensure_int32_contiguous(active_indices.reshape(-1, active_indices.size(-1)))
        count_flat = _ensure_int32_contiguous(active_counts.reshape(-1))
        weight_c = weight.contiguous()
        bias_c = bias.contiguous() if bias is not None else weight.new_empty(0)
        output = run_mps_sparse_projection(x_flat, idx_flat, count_flat, weight_c, bias_c)
        if output is None:
            raise RuntimeError("native sparse projection MPS extension is unavailable")
        return output.view(x.size(0), x.size(1), weight.size(0))

    bias_tensor = bias if bias is not None else weight.new_empty(0)
    return SparseProjectionMPSFunction.apply(
        x,
        selection_mask,
        active_indices,
        active_counts,
        weight,
        bias_tensor,
    )


def native_sparse_project_hidden_pair(
    x: torch.Tensor,
    selection_mask: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    weight_a: torch.Tensor,
    bias_a: Optional[torch.Tensor],
    weight_b: torch.Tensor,
    bias_b: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    requires_grad = (
        torch.is_grad_enabled()
        and (
            x.requires_grad
            or weight_a.requires_grad
            or weight_b.requires_grad
            or (bias_a is not None and bias_a.requires_grad)
            or (bias_b is not None and bias_b.requires_grad)
        )
    )
    if not requires_grad:
        x_flat = x.reshape(-1, x.size(-1)).contiguous()
        idx_flat = _ensure_int32_contiguous(active_indices.reshape(-1, active_indices.size(-1)))
        count_flat = _ensure_int32_contiguous(active_counts.reshape(-1))
        weight_a_c = weight_a.contiguous()
        bias_a_c = bias_a.contiguous() if bias_a is not None else weight_a.new_empty(0)
        weight_b_c = weight_b.contiguous()
        bias_b_c = bias_b.contiguous() if bias_b is not None else weight_b.new_empty(0)
        outputs = run_mps_sparse_projection_pair(
            x_flat, idx_flat, count_flat, weight_a_c, bias_a_c, weight_b_c, bias_b_c
        )
        if outputs is None:
            raise RuntimeError("native sparse projection pair MPS extension is unavailable")
        output_a, output_b = outputs
        return (
            output_a.view(x.size(0), x.size(1), weight_a.size(0)),
            output_b.view(x.size(0), x.size(1), weight_b.size(0)),
        )

    bias_a_tensor = bias_a if bias_a is not None else weight_a.new_empty(0)
    bias_b_tensor = bias_b if bias_b is not None else weight_b.new_empty(0)
    return SparseProjectionPairMPSFunction.apply(
        x,
        selection_mask,
        active_indices,
        active_counts,
        weight_a,
        bias_a_tensor,
        weight_b,
        bias_b_tensor,
    )
