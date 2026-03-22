from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    from dim_gate import HiddenGateOutput
    from mla_native import (
        MAX_NATIVE_ACTIVE,
        mps_sparse_projection_available,
        native_sparse_project_hidden,
        native_sparse_project_hidden_pair,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .dim_gate import HiddenGateOutput
    from .mla_native import (
        MAX_NATIVE_ACTIVE,
        mps_sparse_projection_available,
        native_sparse_project_hidden,
        native_sparse_project_hidden_pair,
    )


# Prefer the native path for sparse projection on supported MPS shapes. It is still slower
# than dense GEMM on the current toy sizes, but it is substantially faster than the padded
# compact sparse fallback that `use_sparse_projection=true` would otherwise use.
PREFER_NATIVE_MPS_SPARSE_PROJECTION = True


def _masked_dense_projection(
    x: torch.Tensor,
    selection_mask: torch.Tensor,
    proj: nn.Module,
) -> torch.Tensor:
    masked_x = x * selection_mask.to(dtype=x.dtype)
    return proj(masked_x)


def _masked_dense_projection_pair(
    x: torch.Tensor,
    selection_mask: torch.Tensor,
    proj_a: nn.Module,
    proj_b: nn.Module,
):
    masked_x = x * selection_mask.to(dtype=x.dtype)
    return proj_a(masked_x), proj_b(masked_x)


def _supports_compact_projection(proj: nn.Module) -> bool:
    weight = getattr(proj, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor):
        return False
    return weight.element_size() > 1


def _supports_native_projection(
    x: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    proj: nn.Module,
) -> bool:
    if x.device.type != "mps" or x.dtype != torch.float32 or not x.is_contiguous():
        return False
    if active_indices.device.type != "mps" or active_counts.device.type != "mps":
        return False
    if not active_indices.is_contiguous() or not active_counts.is_contiguous():
        return False
    if active_indices.numel() == 0:
        return False
    if active_indices.size(-1) > MAX_NATIVE_ACTIVE:
        return False
    if active_counts.dtype not in (torch.int32, torch.int64):
        return False
    if not _supports_compact_projection(proj):
        return False
    if getattr(proj.weight, "device", None) is None or proj.weight.device.type != "mps":
        return False
    if proj.weight.dtype != torch.float32 or not proj.weight.is_contiguous():
        return False
    bias = getattr(proj, "bias", None)
    if bias is not None and (bias.device.type != "mps" or bias.dtype != torch.float32 or not bias.is_contiguous()):
        return False
    return mps_sparse_projection_available()


def _supports_native_projection_pair(
    x: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    proj_a: nn.Module,
    proj_b: nn.Module,
) -> bool:
    return _supports_native_projection(x, active_indices, active_counts, proj_a) and _supports_native_projection(
        x, active_indices, active_counts, proj_b
    )


def _compact_project_hidden(
    x: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    proj: nn.Module,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected 3D projection input, got {tuple(x.shape)}")
    if active_indices.dim() != 3 or active_counts.dim() != 2:
        raise ValueError(
            f"expected active indices/counts with shapes [batch, seq, k] and [batch, seq], got "
            f"{tuple(active_indices.shape)} and {tuple(active_counts.shape)}"
        )

    bsz, seqlen, dim = x.shape
    if active_indices.shape[:2] != (bsz, seqlen) or active_counts.shape != (bsz, seqlen):
        raise ValueError(
            f"gate metadata batch/sequence shape does not match projection input: "
            f"{tuple(active_indices.shape[:2])}, {tuple(active_counts.shape)} vs {(bsz, seqlen)}"
        )

    weight = proj.weight
    out_features = weight.size(0)
    flat_x = x.reshape(-1, dim)
    flat_counts = active_counts.reshape(-1)
    flat_indices = active_indices.reshape(flat_x.size(0), -1)
    max_active = flat_indices.size(-1)

    if max_active == 0:
        output = flat_x.new_zeros(flat_x.size(0), out_features)
        bias = getattr(proj, "bias", None)
        if bias is not None:
            output = output + bias.to(dtype=output.dtype, device=output.device)
        return output.view(bsz, seqlen, out_features)

    safe_indices = flat_indices.clamp(min=0)
    gathered_x = torch.gather(flat_x, 1, safe_indices)

    valid_mask = (
        torch.arange(max_active, device=x.device, dtype=torch.long).unsqueeze(0) < flat_counts.unsqueeze(1)
    )
    gathered_x = gathered_x * valid_mask.to(dtype=x.dtype)

    weight_t = weight.transpose(0, 1).contiguous()
    gathered_weight = weight_t.index_select(0, safe_indices.reshape(-1)).view(flat_x.size(0), max_active, out_features)
    output = torch.bmm(gathered_x.unsqueeze(1), gathered_weight).squeeze(1)

    bias = getattr(proj, "bias", None)
    if bias is not None:
        output = output + bias.to(dtype=output.dtype, device=output.device)
    return output.view(bsz, seqlen, out_features)


def sparse_project_hidden(
    x: torch.Tensor,
    gate_output: Optional[HiddenGateOutput],
    proj: nn.Module,
) -> torch.Tensor:
    if gate_output is None:
        return proj(x)

    selection_mask = gate_output.selection_mask
    if selection_mask.shape != x.shape:
        raise ValueError(
            f"gate selection mask shape {tuple(selection_mask.shape)} does not match projection input {tuple(x.shape)}"
        )

    active_indices = gate_output.active_indices
    active_counts = gate_output.active_counts
    native_active_indices = gate_output.native_active_indices
    native_active_counts = gate_output.native_active_counts
    if (
        active_indices is None
        or active_counts is None
        or not _supports_compact_projection(proj)
    ):
        return _masked_dense_projection(x, selection_mask, proj)

    if PREFER_NATIVE_MPS_SPARSE_PROJECTION and _supports_native_projection(
        x,
        native_active_indices if native_active_indices is not None else active_indices,
        native_active_counts if native_active_counts is not None else active_counts,
        proj,
    ):
        native_indices = native_active_indices if native_active_indices is not None else active_indices.contiguous()
        native_counts = native_active_counts if native_active_counts is not None else active_counts.contiguous()
        return native_sparse_project_hidden(
            x,
            selection_mask,
            native_indices,
            native_counts,
            proj.weight,
            getattr(proj, "bias", None),
        )

    # Future extension point: replace the padded gather with a fully packed/specialized kernel path.
    return _compact_project_hidden(x, active_indices, active_counts, proj)


def sparse_project_hidden_pair(
    x: torch.Tensor,
    gate_output: Optional[HiddenGateOutput],
    proj_a: nn.Module,
    proj_b: nn.Module,
):
    if gate_output is None:
        return proj_a(x), proj_b(x)

    selection_mask = gate_output.selection_mask
    if selection_mask.shape != x.shape:
        raise ValueError(
            f"gate selection mask shape {tuple(selection_mask.shape)} does not match projection input {tuple(x.shape)}"
        )

    active_indices = gate_output.active_indices
    active_counts = gate_output.active_counts
    native_active_indices = gate_output.native_active_indices
    native_active_counts = gate_output.native_active_counts
    if (
        active_indices is None
        or active_counts is None
        or not _supports_compact_projection(proj_a)
        or not _supports_compact_projection(proj_b)
    ):
        return _masked_dense_projection_pair(x, selection_mask, proj_a, proj_b)

    native_indices = native_active_indices if native_active_indices is not None else active_indices
    native_counts = native_active_counts if native_active_counts is not None else active_counts
    if PREFER_NATIVE_MPS_SPARSE_PROJECTION and _supports_native_projection_pair(
        x,
        native_indices,
        native_counts,
        proj_a,
        proj_b,
    ):
        return native_sparse_project_hidden_pair(
            x,
            selection_mask,
            native_indices.contiguous(),
            native_counts.contiguous(),
            proj_a.weight,
            getattr(proj_a, "bias", None),
            proj_b.weight,
            getattr(proj_b, "bias", None),
        )

    return (
        _compact_project_hidden(x, active_indices, active_counts, proj_a),
        _compact_project_hidden(x, active_indices, active_counts, proj_b),
    )
