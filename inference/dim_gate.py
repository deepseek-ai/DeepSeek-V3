import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn


@dataclass
class HiddenGateStats:
    tokens: int = 0
    active_dims: int = 0
    min_active: int = 2**31 - 1
    max_active: int = 0
    raw_active_dims: int = 0
    raw_min_active: int = 2**31 - 1
    raw_max_active: int = 0
    zero_tokens: int = 0
    clamped_min_tokens: int = 0
    clamped_max_tokens: int = 0

    def record(
        self,
        kept: int,
        raw_len: int,
        zero: bool,
        clamped_min: bool,
        clamped_max: bool,
    ) -> None:
        self.tokens += 1
        self.active_dims += kept
        self.min_active = min(self.min_active, kept)
        self.max_active = max(self.max_active, kept)
        self.raw_active_dims += raw_len
        self.raw_min_active = min(self.raw_min_active, raw_len)
        self.raw_max_active = max(self.raw_max_active, raw_len)
        if zero:
            self.zero_tokens += 1
        if clamped_min:
            self.clamped_min_tokens += 1
        if clamped_max:
            self.clamped_max_tokens += 1

    def summary(self) -> dict:
        return {
            "tokens": self.tokens,
            "avg_active_dims": 0.0 if self.tokens == 0 else self.active_dims / self.tokens,
            "min_active": 0 if self.tokens == 0 else self.min_active,
            "max_active": self.max_active,
            "avg_raw_active_dims": 0.0 if self.tokens == 0 else self.raw_active_dims / self.tokens,
            "raw_min_active": 0 if self.tokens == 0 else self.raw_min_active,
            "raw_max_active": self.raw_max_active,
            "zero_tokens": self.zero_tokens,
            "clamped_min_tokens": self.clamped_min_tokens,
            "clamped_max_tokens": self.clamped_max_tokens,
        }


def gate_bounds(d_model: int, gate_frac: float, gate_softness: float) -> Tuple[int, int]:
    if gate_frac >= 0.999:
        return d_model, d_model
    requested = math.floor(d_model * max(0.0, min(1.0, gate_frac)) + 0.5)
    target = max(1, min(d_model, requested))
    softness = max(0.0, min(1.0, gate_softness))
    min_active = math.floor(target * (1.0 - softness))
    max_active = math.ceil(target * (1.0 + softness))
    return max(1, min(d_model, min_active)), max(target, min(d_model, max_active))


class PlasticContextSpiker(nn.Module):
    """
    Inference-time port of the Kanae-Kocho selector dynamics.

    We keep the exact hidden-state evolution, threshold adaptation, context
    accumulation, and learned per-dimension weights. Reward-based update hooks
    are intentionally not wired here because this DeepSeek repo is inference-first.
    """

    def __init__(
        self,
        dim: int,
        max_batch_size: int,
        leak: float,
        threshold: float,
        context_leak: float,
        context_scale: float,
        stdp_lr: float,
        weight_decay: float,
        w_min: float,
        w_max: float,
        target_rate: float,
        threshold_lr: float,
        trace_decay: float,
        active_bonus: float,
        negative_reward_scale: float,
        reward_ema_decay: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_batch_size = max_batch_size
        self.leak = leak
        self.context_leak = context_leak
        self.context_scale = context_scale
        self.stdp_lr = stdp_lr
        self.weight_decay = weight_decay
        self.w_min = w_min
        self.w_max = w_max
        self.target_rate = target_rate
        self.threshold_lr = threshold_lr
        self.trace_decay = trace_decay
        self.active_bonus = active_bonus
        self.negative_reward_scale = negative_reward_scale
        self.reward_ema_decay = reward_ema_decay
        self.initial_threshold = threshold

        self.weights = nn.Parameter(torch.ones(dim))
        self.register_buffer("u", torch.zeros(max_batch_size, dim))
        self.register_buffer("context", torch.zeros(max_batch_size, dim))
        self.register_buffer("trace", torch.zeros(max_batch_size, dim))
        self.register_buffer("threshold", torch.full((max_batch_size,), threshold))

    def reset_state(self, batch_size: int) -> None:
        self.u[:batch_size].zero_()
        self.context[:batch_size].zero_()
        self.trace[:batch_size].zero_()

    def dim_scores(self) -> torch.Tensor:
        return self.weights

    def support_signal(
        self,
        membrane: torch.Tensor,
        combined: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        threshold_scale = torch.maximum(threshold.abs(), torch.ones_like(threshold))
        drive = ((membrane - threshold.unsqueeze(-1)) / threshold_scale.unsqueeze(-1)).clamp_(-4.0, 4.0)
        soft_fire = torch.sigmoid(drive)
        return soft_fire * combined.abs()

    def step(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.dim() != 2 or x.size(-1) != self.dim:
            raise ValueError(f"expected [batch, {self.dim}] input, got {tuple(x.shape)}")
        bsz = x.size(0)
        context = self.context[:bsz]
        membrane = self.u[:bsz]
        trace = self.trace[:bsz]
        threshold = self.threshold[:bsz]

        context.mul_(self.context_leak).add_(x, alpha=1.0 - self.context_leak)
        combined = x + self.context_scale * context
        weighted = combined * self.weights
        membrane.mul_(self.leak).add_(weighted)

        support = self.support_signal(membrane, combined, threshold)
        trace.mul_(self.trace_decay).add_(support)

        active = membrane > threshold.unsqueeze(-1)
        trace.add_(support * active.to(support.dtype) * self.active_bonus)

        rate = active.to(dtype=x.dtype).mean(dim=-1)
        threshold.add_(self.threshold_lr * (rate - self.target_rate)).clamp_(min=0.0)

        return [torch.nonzero(row, as_tuple=False).flatten() for row in active]


class HiddenDimensionGate(nn.Module):
    """
    Dense hidden-dimension masking that preserves the original Kanae selector
    dynamics and clamp logic.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.dim = args.dim
        self.gate_frac = args.gate_frac
        self.gate_softness = args.gate_softness
        self.spiker = PlasticContextSpiker(
            dim=args.dim,
            max_batch_size=args.max_batch_size,
            leak=args.gate_leak,
            threshold=args.gate_threshold,
            context_leak=args.context_leak,
            context_scale=args.context_scale,
            stdp_lr=args.gate_stdp_lr,
            weight_decay=args.gate_weight_decay,
            w_min=args.gate_w_min,
            w_max=args.gate_w_max,
            target_rate=args.gate_target_rate,
            threshold_lr=args.gate_threshold_lr,
            trace_decay=args.gate_trace_decay,
            active_bonus=args.gate_active_bonus,
            negative_reward_scale=args.gate_negative_reward_scale,
            reward_ema_decay=args.gate_reward_ema_decay,
        )
        self.last_stats = HiddenGateStats()

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.dim:
            raise ValueError(f"expected [batch, seq, {self.dim}] input, got {tuple(x.shape)}")

        bsz, seqlen, dim = x.shape
        if start_pos == 0:
            self.spiker.reset_state(bsz)

        if self.gate_frac >= 0.999:
            stats = HiddenGateStats()
            for _ in range(bsz * seqlen):
                stats.record(dim, dim, False, False, False)
            self.last_stats = stats
            return x

        min_active, max_active = gate_bounds(self.dim, self.gate_frac, self.gate_softness)
        mask = torch.zeros_like(x)
        stats = HiddenGateStats()
        weight_scores = self.spiker.dim_scores().abs().to(device=x.device, dtype=x.dtype)

        for t in range(seqlen):
            values = x[:, t, :]
            raw_idx_list = self.spiker.step(values)
            for b, raw_idx in enumerate(raw_idx_list):
                idx = raw_idx
                raw_len = int(idx.numel())
                zero = raw_len == 0
                clamped_min = False
                clamped_max = False

                if raw_len > max_active:
                    idx = idx[:max_active]
                    clamped_max = True

                if idx.numel() < min_active:
                    clamped_min = True
                    seen = torch.zeros(dim, dtype=torch.bool, device=x.device)
                    if idx.numel() > 0:
                        seen[idx] = True
                    fill_scores = values[b].abs() * weight_scores
                    fill_scores = fill_scores.masked_fill(seen, float("-inf"))
                    needed = min_active - idx.numel()
                    extra = torch.topk(fill_scores, k=needed).indices
                    idx = torch.cat([idx, extra], dim=0)

                mask[b, t, idx] = 1
                stats.record(
                    kept=int(idx.numel()),
                    raw_len=raw_len,
                    zero=zero,
                    clamped_min=clamped_min,
                    clamped_max=clamped_max,
                )

        self.last_stats = stats
        return x * mask
