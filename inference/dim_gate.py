import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

try:
    from gate_native import run_mps_gate_scan
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .gate_native import run_mps_gate_scan


IndexSpec = Union[torch.Tensor, Sequence[int]]


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

    def merge(self, other: "HiddenGateStats") -> None:
        if other.tokens == 0:
            return
        self.tokens += other.tokens
        self.active_dims += other.active_dims
        self.min_active = min(self.min_active, other.min_active)
        self.max_active = max(self.max_active, other.max_active)
        self.raw_active_dims += other.raw_active_dims
        self.raw_min_active = min(self.raw_min_active, other.raw_min_active)
        self.raw_max_active = max(self.raw_max_active, other.raw_max_active)
        self.zero_tokens += other.zero_tokens
        self.clamped_min_tokens += other.clamped_min_tokens
        self.clamped_max_tokens += other.clamped_max_tokens

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


@dataclass
class GateOutcome:
    selection_mask: torch.Tensor
    raw_len: int
    zero: bool
    clamped_min: bool
    clamped_max: bool


@dataclass
class HiddenGateOutput:
    selection_mask: torch.Tensor
    stats: HiddenGateStats
    gate_input: torch.Tensor
    active_indices: Optional[torch.Tensor] = None
    active_counts: Optional[torch.Tensor] = None
    native_active_indices: Optional[torch.Tensor] = None
    native_active_counts: Optional[torch.Tensor] = None


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
    Native PyTorch port of Kanae-Kocho's PlasticContextSpiker.

    This keeps the original selector state machine and now also includes the
    reward-driven update hooks from `context_spike.rs` so later training wiring
    can call into the same gate object instead of rebuilding the mechanism.
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
        self.current_batch_size = 0

        self.weights = nn.Parameter(torch.ones(dim))
        self.register_buffer("u", torch.zeros(max_batch_size, dim))
        self.register_buffer("context", torch.zeros(max_batch_size, dim))
        self.register_buffer("trace", torch.zeros(max_batch_size, dim))
        self.register_buffer("threshold", torch.full((max_batch_size,), threshold))
        self.register_buffer("reward_ema", torch.zeros(()))
        self.register_buffer("reward_ema_initialized", torch.tensor(False, dtype=torch.bool))

    def reset_state(self, batch_size: int) -> None:
        self.current_batch_size = batch_size
        self.u[:batch_size].zero_()
        self.context[:batch_size].zero_()
        self.trace[:batch_size].zero_()

    def reset_temporal_state(self, batch_size: Optional[int] = None) -> None:
        self.reset_state(batch_size or self.max_batch_size)

    def dim_scores(self) -> torch.Tensor:
        return self.weights

    @staticmethod
    def _sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def support_signal(
        self,
        membrane: torch.Tensor,
        combined: torch.Tensor,
        threshold: torch.Tensor,
        sanitize: bool = True,
    ) -> torch.Tensor:
        if sanitize:
            membrane = self._sanitize_tensor(membrane)
            combined = self._sanitize_tensor(combined)
            threshold = self._sanitize_tensor(threshold)
        threshold_scale = torch.maximum(threshold.abs(), torch.ones_like(threshold))
        drive = ((membrane - threshold.unsqueeze(-1)) / threshold_scale.unsqueeze(-1)).clamp_(-4.0, 4.0)
        soft_fire = torch.sigmoid(drive)
        return soft_fire * combined.abs()

    def step_row(self, x: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
        return torch.nonzero(self.step_mask_row(x, batch_index), as_tuple=False).flatten()

    def step_mask(self, x: torch.Tensor, sanitize: bool = True) -> torch.Tensor:
        if x.dim() != 2 or x.size(-1) != self.dim:
            raise ValueError(f"expected [batch, {self.dim}] input, got {tuple(x.shape)}")
        if x.size(0) > self.max_batch_size:
            raise ValueError(f"batch size {x.size(0)} exceeds max batch {self.max_batch_size}")

        bsz = x.size(0)
        self.current_batch_size = max(self.current_batch_size, bsz)
        if x.dtype != self.weights.dtype:
            x = x.to(dtype=self.weights.dtype)
        if sanitize:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        context = self.context[:bsz]
        membrane = self.u[:bsz]
        trace = self.trace[:bsz]
        threshold = self.threshold[:bsz]

        context.mul_(self.context_leak).add_(x, alpha=1.0 - self.context_leak)
        combined = x + self.context_scale * context
        weighted = combined * self.weights
        membrane.mul_(self.leak).add_(weighted)

        support = self.support_signal(membrane, combined, threshold, sanitize=False)
        trace.mul_(self.trace_decay).add_(support)

        active = membrane > threshold.unsqueeze(-1)
        trace.add_(support * active.to(support.dtype) * self.active_bonus)

        rate = active.to(dtype=x.dtype).mean(dim=-1)
        threshold.add_(self.threshold_lr * (rate - self.target_rate)).clamp_(min=0.0)
        return active

    def step_mask_row(self, x: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
        if x.dim() != 1 or x.numel() != self.dim:
            raise ValueError(f"expected [{self.dim}] input, got {tuple(x.shape)}")
        if batch_index < 0 or batch_index >= self.max_batch_size:
            raise ValueError(f"batch index {batch_index} is out of range for max batch {self.max_batch_size}")

        if batch_index == 0:
            return self.step_mask(x.unsqueeze(0))[0]

        x = torch.nan_to_num(x.to(dtype=self.weights.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        context = self.context[batch_index]
        membrane = self.u[batch_index]
        trace = self.trace[batch_index]
        threshold = self.threshold[batch_index]

        self.current_batch_size = max(self.current_batch_size, batch_index + 1)
        context.mul_(self.context_leak).add_(x, alpha=1.0 - self.context_leak)
        combined = x + self.context_scale * context
        weighted = combined * self.weights
        membrane.mul_(self.leak).add_(weighted)

        support = self.support_signal(
            membrane.unsqueeze(0),
            combined.unsqueeze(0),
            threshold.view(1),
            sanitize=False,
        )[0]
        trace.mul_(self.trace_decay).add_(support)

        active = membrane > threshold
        trace.add_(support * active.to(support.dtype) * self.active_bonus)

        rate = active.to(dtype=x.dtype).mean()
        threshold.add_(self.threshold_lr * (rate - self.target_rate)).clamp_(min=0.0)
        return active

    def step(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.dim() != 2 or x.size(-1) != self.dim:
            raise ValueError(f"expected [batch, {self.dim}] input, got {tuple(x.shape)}")
        active = self.step_mask(x)
        return [torch.nonzero(row, as_tuple=False).flatten() for row in active]

    def select_dims(self, values: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
        return self.step_row(values, batch_index)

    def select_mask(self, values: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
        return self.step_mask_row(values, batch_index)

    def _normalize_values(self, values: Union[torch.Tensor, Sequence[float]]) -> torch.Tensor:
        values_t = torch.as_tensor(values, device=self.weights.device, dtype=self.weights.dtype)
        values_t = self._sanitize_tensor(values_t)
        if values_t.dim() == 1:
            values_t = values_t.unsqueeze(0)
        if values_t.dim() != 2 or values_t.size(-1) != self.dim:
            raise ValueError(f"expected values with shape [batch, {self.dim}], got {tuple(values_t.shape)}")
        return values_t

    def _normalize_active_dims(
        self,
        active_dims: Union[IndexSpec, Sequence[IndexSpec]],
        batch_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        if batch_size == 1:
            single = active_dims
            if isinstance(active_dims, Sequence) and not isinstance(active_dims, torch.Tensor) and len(active_dims) == 1:
                single = active_dims[0]
            tensor = torch.as_tensor(single, device=device, dtype=torch.long)
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 1:
                return [tensor]

        if not isinstance(active_dims, Sequence) or len(active_dims) != batch_size:
            raise ValueError("active_dims must be a batch-aligned sequence of index tensors")

        out: List[torch.Tensor] = []
        for dims in active_dims:
            tensor = torch.as_tensor(dims, device=device, dtype=torch.long)
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            out.append(tensor.flatten())
        return out

    def _normalize_reward(self, reward: Union[float, torch.Tensor, Sequence[float]], batch_size: int) -> torch.Tensor:
        reward_t = torch.as_tensor(reward, device=self.weights.device, dtype=self.weights.dtype)
        if reward_t.dim() == 0:
            reward_t = reward_t.expand(batch_size)
        elif reward_t.dim() != 1:
            raise ValueError("reward must be a scalar or a 1D batch tensor")
        elif reward_t.numel() == 1:
            reward_t = reward_t.expand(batch_size)
        elif reward_t.numel() != batch_size:
            raise ValueError(f"reward batch has size {reward_t.numel()}, expected {batch_size}")
        return reward_t

    def refresh_trace_from_values(
        self,
        active_dims: Union[IndexSpec, Sequence[IndexSpec]],
        values: Union[torch.Tensor, Sequence[float]],
    ) -> None:
        values_t = self._normalize_values(values)
        batch_size = values_t.size(0)
        self.current_batch_size = max(self.current_batch_size, batch_size)
        active_list = self._normalize_active_dims(active_dims, batch_size, values_t.device)

        for batch_index in range(batch_size):
            active_mask = torch.zeros(self.dim, dtype=torch.bool, device=values_t.device)
            if active_list[batch_index].numel() > 0:
                active_mask[active_list[batch_index].clamp(min=0, max=self.dim - 1)] = True

            combined = values_t[batch_index] + self.context_scale * self.context[batch_index]
            support = self.support_signal(
                self.u[batch_index : batch_index + 1],
                combined.unsqueeze(0),
                self.threshold[batch_index : batch_index + 1],
            )[0]
            support = support + active_mask.to(support.dtype) * self.active_bonus * combined.abs()
            self.trace[batch_index].mul_(self.trace_decay).add_(support)

    def advantage(self, reward: Union[float, torch.Tensor, Sequence[float]]) -> torch.Tensor:
        batch_size = self.current_batch_size if self.current_batch_size > 0 else 1
        reward_t = self._normalize_reward(reward, batch_size)
        mean_reward = reward_t.mean()

        if not bool(self.reward_ema_initialized.item()):
            self.reward_ema.copy_(mean_reward)
            self.reward_ema_initialized.fill_(True)
            return torch.zeros_like(reward_t)

        baseline = self.reward_ema.to(dtype=reward_t.dtype)
        self.reward_ema.mul_(self.reward_ema_decay).add_(mean_reward, alpha=1.0 - self.reward_ema_decay)
        return reward_t - baseline

    def _scaled_advantage(self, reward: Union[float, torch.Tensor, Sequence[float]], batch_size: int) -> torch.Tensor:
        advantage = self.advantage(reward)
        if advantage.numel() != batch_size:
            advantage = self._normalize_reward(advantage, batch_size)
        return torch.where(advantage < 0.0, advantage * self.negative_reward_scale, advantage)

    def _apply_weight_delta(self, delta: torch.Tensor) -> None:
        with torch.no_grad():
            self.weights.add_(self.stdp_lr * delta)
            self.weights.mul_(1.0 - self.weight_decay)
            self.weights.clamp_(self.w_min, self.w_max)
            self.recenter_weights()

    def apply_reward(self, reward: Union[float, torch.Tensor, Sequence[float]]) -> None:
        batch_size = self.current_batch_size if self.current_batch_size > 0 else 1
        reward_t = torch.as_tensor(reward, device=self.weights.device, dtype=self.weights.dtype)
        if not torch.isfinite(reward_t).any():
            self.clear_trace()
            return
        reward_t = self._sanitize_tensor(reward_t)
        scaled_advantage = self._scaled_advantage(reward_t, batch_size)
        delta = (self.trace[:batch_size] * scaled_advantage.unsqueeze(-1)).mean(dim=0)
        self._apply_weight_delta(delta)
        self.clear_trace()

    def apply_token_reward(
        self,
        active_dims: Union[IndexSpec, Sequence[IndexSpec]],
        values: Union[torch.Tensor, Sequence[float]],
        reward: Union[float, torch.Tensor, Sequence[float]],
    ) -> None:
        values_t = self._normalize_values(values)
        batch_size = values_t.size(0)
        reward_t = torch.as_tensor(reward, device=self.weights.device, dtype=self.weights.dtype)
        if not torch.isfinite(reward_t).any():
            self.clear_trace()
            return
        reward_t = self._sanitize_tensor(reward_t)
        self.refresh_trace_from_values(active_dims, values_t)
        scaled_advantage = self._scaled_advantage(reward_t, batch_size)
        delta = (self.trace[:batch_size] * scaled_advantage.unsqueeze(-1)).mean(dim=0)
        self._apply_weight_delta(delta)
        self.clear_trace()

    def apply_batch_signal(self, signal: Union[torch.Tensor, Sequence[float]], cutoff: float) -> None:
        self.apply_batch_signal_scaled(signal, cutoff, 1.0)

    def apply_batch_signal_scaled(
        self,
        signal: Union[torch.Tensor, Sequence[float]],
        cutoff: float,
        scale: float,
    ) -> None:
        signal_t = self._sanitize_tensor(
            torch.as_tensor(signal, device=self.weights.device, dtype=self.weights.dtype).flatten()
        )
        n = min(self.weights.numel(), signal_t.numel())
        if n == 0:
            self.clear_trace()
            return
        scale = max(0.0, scale)
        if scale <= torch.finfo(self.weights.dtype).eps:
            self.clear_trace()
            return

        signal_t = signal_t[:n]
        mean_signal = signal_t.mean()
        max_abs = signal_t.abs().amax().clamp_min(1e-6)
        delta = signal_t - mean_signal
        confidence = signal_t.abs() / max_abs
        update_mask = confidence >= cutoff
        scaled = torch.where(delta < 0.0, delta * self.negative_reward_scale, delta)
        scaled = torch.where(update_mask, scaled * scale, torch.zeros_like(scaled))

        full_delta = torch.zeros_like(self.weights)
        full_delta[:n] = scaled
        self._apply_weight_delta(full_delta)
        self.clear_trace()

    def apply_batch_signal_soft(
        self,
        signal: Union[torch.Tensor, Sequence[float]],
        cutoff: float,
        power: float,
    ) -> None:
        signal_t = self._sanitize_tensor(
            torch.as_tensor(signal, device=self.weights.device, dtype=self.weights.dtype).flatten()
        )
        n = min(self.weights.numel(), signal_t.numel())
        if n == 0:
            self.clear_trace()
            return

        signal_t = signal_t[:n]
        mean_signal = signal_t.mean()
        max_abs = signal_t.abs().amax().clamp_min(1e-6)
        delta = signal_t - mean_signal
        confidence = signal_t.abs() / max_abs
        power = max(0.0, power)
        lr_scale = torch.ones_like(confidence) if power == 0.0 else confidence.pow(power)
        update_mask = confidence >= cutoff
        scaled = torch.where(delta < 0.0, delta * self.negative_reward_scale, delta)
        scaled = torch.where(update_mask, scaled * lr_scale, torch.zeros_like(scaled))

        full_delta = torch.zeros_like(self.weights)
        full_delta[:n] = scaled
        self._apply_weight_delta(full_delta)
        self.clear_trace()

    def plasticity_scores(self, values: Union[torch.Tensor, Sequence[float]]) -> torch.Tensor:
        values_t = self._normalize_values(values)
        batch_size = values_t.size(0)
        combined = values_t + self.context_scale * self.context[:batch_size]
        scores = self.support_signal(self.u[:batch_size], combined, self.threshold[:batch_size])

        weight_span = max(self.w_max - self.w_min, 1e-6)
        weight_conf = ((self.weights - self.w_min) / weight_span).clamp_(0.0, 1.0)
        scores = scores * (0.5 + 0.5 * weight_conf.unsqueeze(0))

        active = self.u[:batch_size] > self.threshold[:batch_size].unsqueeze(-1)
        active_scale = 1.0 + 0.5 * self.active_bonus * active.to(scores.dtype)
        scores = scores * active_scale
        scores = self._sanitize_tensor(scores)
        scores = scores / scores.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return scores[0] if values_t.size(0) == 1 else scores

    def recenter_weights(self) -> None:
        with torch.no_grad():
            if self.weights.numel() == 0:
                return
            self.weights.copy_(self._sanitize_tensor(self.weights))
            shift = self.weights.mean() - 1.0
            self.weights.sub_(shift).clamp_(self.w_min, self.w_max)

    def clear_trace(self) -> None:
        self.trace.zero_()


class HiddenDimensionGate(nn.Module):
    """
    Hidden-dimension gate with the core `hidden_gate.rs` modes ported into the
    DeepSeek inference stack.

    The forward path still produces a dense mask, but the gate now also exposes
    the selector-learning API needed for manual training wiring later.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.dim = args.dim
        self.gate_frac = args.gate_frac
        self.gate_softness = args.gate_softness
        self.shared_basis = getattr(args, "gate_shared_basis", False)
        self.basis_window = int(getattr(args, "gate_basis_window", 0))
        self.force_zero = getattr(args, "gate_force_zero", False)
        self.sort_indices = getattr(args, "gate_sort_indices", False)
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
        self.last_output: Optional[HiddenGateOutput] = None
        self.stats_enabled = True
        self.native_scan_enabled = bool(getattr(args, "gate_native_scan", True))
        self.return_gate_metadata = bool(getattr(args, "return_gate_metadata", False))
        self.register_buffer("shared_mean", torch.zeros(args.max_batch_size, args.dim))
        self.register_buffer("shared_count", torch.zeros(args.max_batch_size, dtype=torch.long))
        self.register_buffer("basis_pos", torch.zeros(args.max_batch_size, dtype=torch.long))
        self._history: List[List[torch.Tensor]] = [[] for _ in range(args.max_batch_size)]
        self._basis_outcome: List[Optional[GateOutcome]] = [None for _ in range(args.max_batch_size)]

    def reset_temporal_state(self, batch_size: int) -> None:
        self.spiker.reset_state(batch_size)
        self.shared_mean[:batch_size].zero_()
        self.shared_count[:batch_size].zero_()
        self.basis_pos[:batch_size].zero_()
        for batch_index in range(batch_size):
            self._history[batch_index] = []
            self._basis_outcome[batch_index] = None
        self.last_output = None

    def _record_tokens(self, stats: HiddenGateStats, outcome: GateOutcome, tokens: int) -> None:
        kept = int(outcome.selection_mask.sum().item())
        for _ in range(tokens):
            stats.record(
                kept=kept,
                raw_len=outcome.raw_len,
                zero=outcome.zero,
                clamped_min=outcome.clamped_min,
                clamped_max=outcome.clamped_max,
            )

    def _basis_advance(self, batch_index: int) -> bool:
        if self.basis_window <= 0:
            return True
        refresh = int(self.basis_pos[batch_index].item()) == 0
        self.basis_pos[batch_index] = (self.basis_pos[batch_index] + 1) % self.basis_window
        return refresh

    def _window_mean(self, batch_index: int, current: torch.Tensor) -> torch.Tensor:
        history = self._history[batch_index]
        if not history:
            return current
        return torch.stack(history[-self.basis_window :], dim=0).mean(dim=0)

    def _update_running_mean(self, batch_index: int, values: torch.Tensor) -> None:
        count = int(self.shared_count[batch_index].item())
        if count <= 0:
            self.shared_mean[batch_index].copy_(values)
            self.shared_count[batch_index] = 1
            return

        new_count = count + 1
        weight_old = count / new_count
        weight_new = 1.0 / new_count
        self.shared_mean[batch_index].mul_(weight_old).add_(values, alpha=weight_new)
        self.shared_count[batch_index] = new_count

    def _clamp_active_mask(
        self,
        raw_mask: torch.Tensor,
        values: torch.Tensor,
        weight_scores: torch.Tensor,
        min_active: int,
        max_active: int,
    ) -> GateOutcome:
        if self.force_zero:
            return GateOutcome(
                selection_mask=torch.zeros(values.numel(), dtype=torch.bool, device=values.device),
                raw_len=0,
                zero=True,
                clamped_min=False,
                clamped_max=False,
            )

        selection_mask = raw_mask.flatten().to(dtype=torch.bool)
        raw_len = int(selection_mask.sum().item())
        zero = raw_len == 0
        clamped_min = False
        clamped_max = False

        if raw_len > max_active:
            order = selection_mask.to(dtype=torch.int32).cumsum(dim=0)
            selection_mask = selection_mask & (order <= max_active)
            clamped_max = True
        elif raw_len < min_active:
            clamped_min = True
            fill_scores = values.abs() * weight_scores
            fill_scores = fill_scores.masked_fill(selection_mask, float("-inf"))
            needed = min_active - raw_len
            extra = torch.topk(fill_scores, k=needed).indices
            selection_mask = selection_mask.clone()
            selection_mask[extra] = True

        return GateOutcome(
            selection_mask=selection_mask,
            raw_len=raw_len,
            zero=zero,
            clamped_min=clamped_min,
            clamped_max=clamped_max,
        )

    def _clamp_active_batch(
        self,
        raw_mask: torch.Tensor,
        values: torch.Tensor,
        weight_scores: torch.Tensor,
        min_active: int,
        max_active: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.force_zero:
            zeros = torch.zeros_like(raw_mask, dtype=torch.bool)
            counts = torch.zeros(raw_mask.size(0), device=raw_mask.device, dtype=torch.long)
            flags = torch.zeros_like(counts, dtype=torch.bool)
            return zeros, counts, counts, flags, flags, flags

        selection_mask = raw_mask.to(dtype=torch.bool)
        raw_counts = selection_mask.sum(dim=-1)
        clamped_min = raw_counts < min_active
        clamped_max = raw_counts > max_active

        order = selection_mask.to(dtype=torch.int32).cumsum(dim=-1)
        trimmed = selection_mask & (order <= max_active)
        selection_mask = torch.where(clamped_max.unsqueeze(-1), trimmed, selection_mask)

        fill_scores = values.abs() * weight_scores.unsqueeze(0)
        active_priority = fill_scores.amax(dim=-1, keepdim=True).clamp_min(1.0) + 1.0
        priority = torch.where(selection_mask, fill_scores + active_priority, fill_scores)
        top_idx = torch.topk(priority, k=min_active, dim=-1).indices
        fill_mask = torch.zeros_like(selection_mask)
        fill_mask.scatter_(1, top_idx, True)
        selection_mask = torch.where(clamped_min.unsqueeze(-1), fill_mask, selection_mask)

        kept_counts = selection_mask.sum(dim=-1)
        zero_mask = raw_counts == 0
        return selection_mask, raw_counts, kept_counts, zero_mask, clamped_min, clamped_max

    def _merge_batch_stats(
        self,
        stats: HiddenGateStats,
        kept_counts: torch.Tensor,
        raw_counts: torch.Tensor,
        zero_mask: torch.Tensor,
        clamped_min: torch.Tensor,
        clamped_max: torch.Tensor,
    ) -> None:
        if kept_counts.numel() == 0:
            return

        kept_cpu = kept_counts.detach().to(device="cpu", dtype=torch.long)
        raw_cpu = raw_counts.detach().to(device="cpu", dtype=torch.long)
        zero_cpu = zero_mask.detach().to(device="cpu", dtype=torch.long)
        min_cpu = clamped_min.detach().to(device="cpu", dtype=torch.long)
        max_cpu = clamped_max.detach().to(device="cpu", dtype=torch.long)

        stats.tokens += int(kept_cpu.numel())
        stats.active_dims += int(kept_cpu.sum().item())
        stats.min_active = min(stats.min_active, int(kept_cpu.min().item()))
        stats.max_active = max(stats.max_active, int(kept_cpu.max().item()))
        stats.raw_active_dims += int(raw_cpu.sum().item())
        stats.raw_min_active = min(stats.raw_min_active, int(raw_cpu.min().item()))
        stats.raw_max_active = max(stats.raw_max_active, int(raw_cpu.max().item()))
        stats.zero_tokens += int(zero_cpu.sum().item())
        stats.clamped_min_tokens += int(min_cpu.sum().item())
        stats.clamped_max_tokens += int(max_cpu.sum().item())

    def _build_active_metadata(
        self,
        selection_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.return_gate_metadata:
            return None, None

        active_counts = selection_mask.sum(dim=-1, dtype=torch.long)
        if active_counts.numel() == 0:
            empty_indices = torch.full(
                (*selection_mask.shape[:2], 0),
                -1,
                device=selection_mask.device,
                dtype=torch.long,
            )
            return empty_indices, active_counts

        max_active = int(active_counts.max().item())
        if max_active == 0:
            empty_indices = torch.full(
                (*selection_mask.shape[:2], 0),
                -1,
                device=selection_mask.device,
                dtype=torch.long,
            )
            return empty_indices, active_counts

        flat_mask = selection_mask.reshape(-1, self.dim)
        flat_dim_idx = torch.arange(self.dim, device=selection_mask.device, dtype=torch.long)
        flat_dim_idx = flat_dim_idx.unsqueeze(0).expand(flat_mask.size(0), -1)
        positions = flat_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        sentinel = torch.full_like(positions, max_active)
        positions = torch.where(flat_mask, positions, sentinel)
        padded_indices = torch.full(
            (flat_mask.size(0), max_active + 1),
            -1,
            device=selection_mask.device,
            dtype=torch.long,
        )
        padded_indices.scatter_(
            1,
            positions,
            torch.where(flat_mask, flat_dim_idx, torch.zeros_like(flat_dim_idx)),
        )
        active_indices = padded_indices[:, :max_active].reshape(*selection_mask.shape[:2], max_active)
        return active_indices, active_counts

    def _select_full_sequence_shared_basis(
        self,
        sequence: torch.Tensor,
        batch_index: int,
        min_active: int,
        max_active: int,
        weight_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, HiddenGateStats]:
        stats = HiddenGateStats()
        mean = sequence.mean(dim=0)
        raw_mask = self.spiker.select_mask(mean, batch_index)
        outcome = self._clamp_active_mask(raw_mask, mean, weight_scores, min_active, max_active)
        self.shared_mean[batch_index].copy_(mean)
        self.shared_count[batch_index] = sequence.size(0)
        self._basis_outcome[batch_index] = outcome
        self._record_tokens(stats, outcome, sequence.size(0))
        return outcome.selection_mask.unsqueeze(0).expand(sequence.size(0), -1).clone(), stats

    def _select_token_sequence(
        self,
        sequence: torch.Tensor,
        batch_index: int,
        min_active: int,
        max_active: int,
        weight_scores: torch.Tensor,
        collect_stats: bool,
    ) -> Tuple[torch.Tensor, HiddenGateStats]:
        stats = HiddenGateStats()
        selection_rows = torch.zeros(sequence.size(0), self.dim, dtype=torch.bool, device=sequence.device)

        for token_index, token in enumerate(sequence):
            if self.shared_basis and self.basis_window > 0:
                refresh = self._basis_advance(batch_index) or self._basis_outcome[batch_index] is None
                if refresh:
                    mean = self._window_mean(batch_index, token)
                    raw_mask = self.spiker.select_mask(mean, batch_index)
                    self._basis_outcome[batch_index] = self._clamp_active_mask(
                        raw_mask,
                        mean,
                        weight_scores,
                        min_active,
                        max_active,
                    )
                outcome = self._basis_outcome[batch_index]
                if outcome is None:
                    raise RuntimeError("basis outcome was unexpectedly empty")
                self._history[batch_index].append(token.detach().clone())
                if len(self._history[batch_index]) > self.basis_window:
                    self._history[batch_index].pop(0)
            elif self.shared_basis:
                self._update_running_mean(batch_index, token)
                raw_mask = self.spiker.select_mask(self.shared_mean[batch_index], batch_index)
                outcome = self._clamp_active_mask(
                    raw_mask,
                    self.shared_mean[batch_index],
                    weight_scores,
                    min_active,
                    max_active,
                )
                self._basis_outcome[batch_index] = outcome
            else:
                raw_mask = self.spiker.select_mask(token, batch_index)
                outcome = self._clamp_active_mask(raw_mask, token, weight_scores, min_active, max_active)

            selection_rows[token_index] = outcome.selection_mask
            if collect_stats:
                stats.record(
                    kept=int(outcome.selection_mask.sum().item()),
                    raw_len=outcome.raw_len,
                    zero=outcome.zero,
                    clamped_min=outcome.clamped_min,
                    clamped_max=outcome.clamped_max,
                )

        return selection_rows, stats

    def _select_batched_tokens(
        self,
        sequences: torch.Tensor,
        min_active: int,
        max_active: int,
        weight_scores: torch.Tensor,
        collect_stats: bool,
    ) -> Tuple[torch.Tensor, HiddenGateStats]:
        stats = HiddenGateStats()
        _bsz, seqlen, _ = sequences.shape
        selection_mask = torch.zeros_like(sequences, dtype=torch.bool)
        if collect_stats:
            kept_all = torch.zeros(seqlen, _bsz, dtype=torch.long, device=sequences.device)
            raw_all = torch.zeros(seqlen, _bsz, dtype=torch.long, device=sequences.device)
            zero_all = torch.zeros(seqlen, _bsz, dtype=torch.bool, device=sequences.device)
            min_all = torch.zeros(seqlen, _bsz, dtype=torch.bool, device=sequences.device)
            max_all = torch.zeros(seqlen, _bsz, dtype=torch.bool, device=sequences.device)

        for token_index in range(seqlen):
            token_values = sequences[:, token_index, :]
            raw_mask = self.spiker.step_mask(token_values, sanitize=False)
            token_selection, raw_counts, kept_counts, zero_mask, clamped_min, clamped_max = self._clamp_active_batch(
                raw_mask,
                token_values,
                weight_scores,
                min_active,
                max_active,
            )
            selection_mask[:, token_index, :] = token_selection
            if collect_stats:
                kept_all[token_index] = kept_counts
                raw_all[token_index] = raw_counts
                zero_all[token_index] = zero_mask
                min_all[token_index] = clamped_min
                max_all[token_index] = clamped_max

        if collect_stats:
            self._merge_batch_stats(
                stats,
                kept_all.reshape(-1),
                raw_all.reshape(-1),
                zero_all.reshape(-1),
                min_all.reshape(-1),
                max_all.reshape(-1),
            )

        return selection_mask, stats

    def _select_batched_tokens_native(
        self,
        sequences: torch.Tensor,
        min_active: int,
        max_active: int,
        collect_stats: bool,
    ) -> Optional[torch.Tensor]:
        if (
            not self.native_scan_enabled
            or collect_stats
            or self.shared_basis
            or sequences.device.type != "mps"
            or sequences.dtype != torch.float32
            or self.spiker.weights.dtype != torch.float32
        ):
            return None

        if not sequences.is_contiguous():
            sequences = sequences.contiguous()

        bsz = sequences.size(0)
        context = self.spiker.context[:bsz]
        membrane = self.spiker.u[:bsz]
        trace = self.spiker.trace[:bsz]
        threshold = self.spiker.threshold[:bsz]
        weights = self.spiker.weights.detach()

        if not (
            context.is_contiguous()
            and membrane.is_contiguous()
            and trace.is_contiguous()
            and threshold.is_contiguous()
            and weights.is_contiguous()
        ):
            return None

        native_mask = run_mps_gate_scan(
            sequences,
            weights,
            context,
            membrane,
            trace,
            threshold,
            self.spiker.leak,
            self.spiker.context_leak,
            self.spiker.context_scale,
            self.spiker.threshold_lr,
            self.spiker.target_rate,
            self.spiker.trace_decay,
            self.spiker.active_bonus,
            min_active,
            max_active,
        )
        if native_mask is None:
            return None
        return native_mask.to(dtype=torch.bool)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.dim:
            raise ValueError(f"expected [batch, seq, {self.dim}] input, got {tuple(x.shape)}")

        bsz, seqlen, dim = x.shape
        if start_pos == 0:
            self.reset_temporal_state(bsz)
        else:
            self.spiker.current_batch_size = max(self.spiker.current_batch_size, bsz)

        collect_stats = (not self.training) and self.stats_enabled
        with torch.no_grad():
            stats = HiddenGateStats()

            if self.gate_frac >= 0.999 and not self.force_zero:
                selection_mask = torch.ones_like(x, dtype=torch.bool)
                active_indices, active_counts = self._build_active_metadata(selection_mask)
                if collect_stats:
                    for _ in range(bsz * seqlen):
                        stats.record(dim, dim, False, False, False)
                self.last_stats = stats
                self.last_output = HiddenGateOutput(
                    selection_mask=selection_mask,
                    stats=stats,
                    gate_input=x.detach().to(dtype=self.spiker.weights.dtype),
                    active_indices=active_indices,
                    active_counts=active_counts,
                )
                return x

            min_active, max_active = gate_bounds(self.dim, self.gate_frac, self.gate_softness)
            selection_mask = torch.zeros_like(x, dtype=torch.bool)
            weight_scores = self.spiker.dim_scores().abs().to(device=x.device, dtype=self.spiker.weights.dtype)
            x_gate = torch.nan_to_num(
                x.detach().to(dtype=self.spiker.weights.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if not self.shared_basis:
                native_mask = self._select_batched_tokens_native(
                    x_gate,
                    min_active,
                    max_active,
                    collect_stats,
                )
                if native_mask is not None:
                    selection_mask = native_mask
                else:
                    selection_mask, stats = self._select_batched_tokens(
                        x_gate,
                        min_active,
                        max_active,
                        weight_scores,
                        collect_stats,
                    )
            else:
                for batch_index in range(bsz):
                    sequence = x_gate[batch_index]
                    if self.basis_window == 0 and start_pos == 0 and seqlen > 1:
                        batch_idx, batch_stats = self._select_full_sequence_shared_basis(
                            sequence,
                            batch_index,
                            min_active,
                            max_active,
                            weight_scores,
                        )
                    else:
                        batch_selection, batch_stats = self._select_token_sequence(
                            sequence,
                            batch_index,
                            min_active,
                            max_active,
                            weight_scores,
                            collect_stats,
                        )
                        selection_mask[batch_index] = batch_selection
                    if self.basis_window == 0 and start_pos == 0 and seqlen > 1:
                        selection_mask[batch_index] = batch_idx
                    if collect_stats:
                        stats.merge(batch_stats)

        self.last_stats = stats
        active_indices, active_counts = self._build_active_metadata(selection_mask)
        native_active_indices = None
        native_active_counts = None
        if active_indices is not None and active_counts is not None and selection_mask.device.type == "mps":
            native_active_indices = active_indices.to(dtype=torch.int32).contiguous()
            native_active_counts = active_counts.to(dtype=torch.int32).contiguous()
        self.last_output = HiddenGateOutput(
            selection_mask=selection_mask,
            stats=stats,
            gate_input=x_gate,
            active_indices=active_indices,
            active_counts=active_counts,
            native_active_indices=native_active_indices,
            native_active_counts=native_active_counts,
        )
        return x * selection_mask.to(dtype=x.dtype)

    def apply_reward(self, reward: Union[float, torch.Tensor, Sequence[float]]) -> None:
        self.spiker.apply_reward(reward)

    def apply_token_reward(
        self,
        active_dims: Union[IndexSpec, Sequence[IndexSpec]],
        values: Union[torch.Tensor, Sequence[float]],
        reward: Union[float, torch.Tensor, Sequence[float]],
    ) -> None:
        self.spiker.apply_token_reward(active_dims, values, reward)

    def apply_batch_signal(self, signal: Union[torch.Tensor, Sequence[float]], cutoff: float) -> None:
        self.spiker.apply_batch_signal(signal, cutoff)

    def apply_batch_signal_scaled(
        self,
        signal: Union[torch.Tensor, Sequence[float]],
        cutoff: float,
        scale: float,
    ) -> None:
        self.spiker.apply_batch_signal_scaled(signal, cutoff, scale)

    def apply_batch_signal_soft(
        self,
        signal: Union[torch.Tensor, Sequence[float]],
        cutoff: float,
        power: float,
    ) -> None:
        self.spiker.apply_batch_signal_soft(signal, cutoff, power)

    def plasticity_scores(self, values: Union[torch.Tensor, Sequence[float]]) -> torch.Tensor:
        return self.spiker.plasticity_scores(values)

    def apply_token_rewards_from_last(
        self,
        rewards: Union[torch.Tensor, Sequence[float], Sequence[Sequence[float]]],
        start_at: int = 0,
    ) -> None:
        if self.last_output is None:
            raise RuntimeError("no cached gate output is available; run a forward pass first")

        gate_input = self.last_output.gate_input
        selection_mask = self.last_output.selection_mask
        bsz, seqlen, _ = gate_input.shape
        reward_t = torch.as_tensor(rewards, device=gate_input.device, dtype=gate_input.dtype)
        if reward_t.dim() == 0:
            reward_t = reward_t.expand(bsz, seqlen)
        elif reward_t.dim() == 1:
            if reward_t.numel() == seqlen:
                reward_t = reward_t.unsqueeze(0).expand(bsz, seqlen)
            elif reward_t.numel() == bsz:
                reward_t = reward_t.unsqueeze(1).expand(bsz, seqlen)
            else:
                raise ValueError(f"1D rewards must have length {seqlen} or {bsz}")
        elif reward_t.shape != (bsz, seqlen):
            raise ValueError(f"expected rewards with shape [{bsz}, {seqlen}], got {tuple(reward_t.shape)}")

        active_index_cache = self.last_output.active_indices
        for token_index in range(max(0, start_at), seqlen):
            if active_index_cache is not None:
                counts = self.last_output.active_counts[:, token_index]
                active_dims = [
                    active_index_cache[batch_index, token_index, : int(counts[batch_index].item())]
                    for batch_index in range(bsz)
                ]
            else:
                active_dims = [
                    torch.nonzero(selection_mask[batch_index, token_index], as_tuple=False).flatten()
                    for batch_index in range(bsz)
                ]
            values = gate_input[:, token_index, :]
            reward_row = reward_t[:, token_index]
            if not torch.isfinite(reward_row).any():
                continue
            self.spiker.apply_token_reward(active_dims, values, reward_row)
