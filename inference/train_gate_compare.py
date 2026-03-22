import json
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import ModelArgs, Transformer, build_causal_mask


def select_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype_for(device: torch.device, dtype_name: str) -> torch.dtype:
    if dtype_name == "fp32" or device.type != "cuda":
        return torch.float32
    return torch.bfloat16


def load_corpus(paths: Sequence[Path]) -> torch.Tensor:
    data = bytearray()
    for path in paths:
        data.extend(path.read_bytes())
        data.extend(b"\n")
    if len(data) < 1024:
        raise ValueError("corpus is too small for a meaningful compare run")
    return torch.tensor(list(data), dtype=torch.long)


def split_corpus(data: torch.Tensor, train_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
    cutoff = max(512, min(data.numel() - 512, int(data.numel() * train_frac)))
    return data[:cutoff], data[cutoff:]


def make_batches(
    data: torch.Tensor,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> List[torch.Tensor]:
    if data.numel() <= seq_len + 1:
        raise ValueError("corpus split is too small for the requested sequence length")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    max_start = data.numel() - seq_len - 1
    batches: List[torch.Tensor] = []
    for _ in range(num_batches):
        starts = torch.randint(0, max_start, (batch_size,), generator=gen)
        rows = [data[s : s + seq_len + 1] for s in starts.tolist()]
        batches.append(torch.stack(rows, dim=0))
    return batches


def build_model_args(
    base_args: ModelArgs,
    use_gate: bool,
    train_batch_size: int,
    eval_batch_size: int,
    seq_len: int,
    dim: int,
    n_layers: int,
    inter_dim: int,
    n_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    gate_overrides: Optional[Dict[str, object]] = None,
) -> ModelArgs:
    cfg = asdict(base_args)
    cfg.update(
        {
            "dtype": "fp32",
            "vocab_size": 256,
            "max_batch_size": max(train_batch_size, eval_batch_size),
            "max_seq_len": max(seq_len, 32),
            "dim": dim,
            "n_layers": n_layers,
            "n_dense_layers": n_layers,
            "inter_dim": inter_dim,
            "n_heads": n_heads,
            "kv_lora_rank": kv_lora_rank,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "v_head_dim": v_head_dim,
            "use_dim_gate": use_gate,
        }
    )
    if gate_overrides:
        cfg.update(gate_overrides)
    return ModelArgs(**cfg)


def batch_to_inputs_targets(batch: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = batch.to(device=device, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]


def reset_attention_caches(model: Transformer) -> None:
    for layer in model.layers:
        attn = layer.attn
        if hasattr(attn, "k_cache"):
            attn.k_cache.zero_()
            attn.v_cache.zero_()
        if hasattr(attn, "kv_cache"):
            attn.kv_cache.zero_()
            attn.pe_cache.zero_()


def set_gate_stats_enabled(model: Transformer, enabled: bool) -> None:
    for layer in model.layers:
        gate = layer.dim_gate
        if gate is not None:
            gate.stats_enabled = enabled


def forward_all_logits(model: Transformer, tokens: torch.Tensor) -> torch.Tensor:
    seqlen = tokens.size(1)
    reset_attention_caches(model)
    h = model.embed(tokens)
    freqs_cis = model.freqs_cis[:seqlen]
    mask = build_causal_mask(seqlen, tokens.device)
    for layer in model.layers:
        h = layer(h, 0, freqs_cis, mask)
    h = model.norm(h)
    logits = model.head(h)
    return logits


def loss_and_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    vocab = logits.size(-1)
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    preds = logits.argmax(dim=-1)
    accuracy = (preds == targets).float().mean().item()
    return token_loss.mean(), token_loss, accuracy


def capture_gate_snapshot(model: Transformer) -> Dict[int, dict]:
    snapshot: Dict[int, dict] = {}
    for layer_id, layer in enumerate(model.layers):
        gate = layer.dim_gate
        if gate is None or gate.last_output is None:
            continue
        selection_mask = gate.last_output.selection_mask.detach().cpu().clone()
        unique_dims = int(selection_mask.any(dim=(0, 1)).sum().item())
        weights = gate.spiker.weights.detach().cpu()
        snapshot[layer_id] = {
            "selection_mask": selection_mask,
            "unique_dims": unique_dims,
            "weight_mean": float(weights.mean().item()),
            "weight_std": float(weights.std(unbiased=False).item()),
            "weights": weights.clone(),
            "thresholds": gate.spiker.threshold[: max(1, gate.spiker.current_batch_size)].detach().cpu().clone(),
            "stats": gate.last_stats.summary(),
        }
    return snapshot


def compare_snapshots(previous: Optional[Dict[int, dict]], current: Dict[int, dict]) -> Dict[int, dict]:
    metrics: Dict[int, dict] = {}
    for layer_id, layer_data in current.items():
        result = {
            "unique_dims": layer_data["unique_dims"],
            "mean_selection_jaccard": 1.0,
            "selection_change_rate": 0.0,
            "mean_abs_weight_delta": 0.0,
            "max_abs_weight_delta": 0.0,
            "mean_abs_threshold_delta": 0.0,
            "stats": layer_data["stats"],
        }
        if previous is not None and layer_id in previous:
            current_mask = layer_data["selection_mask"].reshape(-1, layer_data["selection_mask"].size(-1))
            previous_mask = previous[layer_id]["selection_mask"].reshape(-1, previous[layer_id]["selection_mask"].size(-1))
            n = min(previous_mask.size(0), current_mask.size(0))
            if n > 0:
                prev_rows = previous_mask[:n]
                cur_rows = current_mask[:n]
                intersection = (prev_rows & cur_rows).sum(dim=-1).to(dtype=torch.float32)
                union = (prev_rows | cur_rows).sum(dim=-1)
                union_f = union.to(dtype=torch.float32)
                jaccard = torch.where(union == 0, torch.ones_like(union_f), intersection / union_f)
                changed = (prev_rows != cur_rows).any(dim=-1).to(dtype=torch.float32)
                result["mean_selection_jaccard"] = float(jaccard.mean().item())
                result["selection_change_rate"] = float(changed.mean().item())
            weight_delta = (layer_data["weights"] - previous[layer_id]["weights"]).abs()
            result["mean_abs_weight_delta"] = float(weight_delta.mean().item())
            result["max_abs_weight_delta"] = float(weight_delta.max().item())
            threshold_delta = (layer_data["thresholds"] - previous[layer_id]["thresholds"]).abs()
            result["mean_abs_threshold_delta"] = float(threshold_delta.mean().item())
        metrics[layer_id] = result
    return metrics


def center_rewards(rewards: torch.Tensor, start_at: int) -> torch.Tensor:
    centered = rewards.clone()
    if start_at >= centered.size(1):
        return centered
    tail = centered[:, start_at:]
    centered[:, start_at:] = tail - tail.mean(dim=1, keepdim=True)
    return centered


def fixed_mask_keep_count(d_model: int, gate_frac: float) -> int:
    return max(1, min(d_model, int(round(d_model * max(0.0, min(1.0, gate_frac))))))


def ranked_batch_signal(update: torch.Tensor, gate_frac: float) -> torch.Tensor:
    if update.numel() == 0:
        return update
    ranked = torch.zeros_like(update)
    band = min(update.numel(), fixed_mask_keep_count(update.numel(), gate_frac))
    order = torch.argsort(update, descending=True)
    margins = torch.arange(band, device=update.device, dtype=update.dtype)
    margins = ((band - margins) / max(1, band)).clamp_min(0.1)
    top_idx = order[:band]
    bottom_idx = torch.flip(order, dims=[0])[:band]
    ranked[top_idx] = torch.clamp_min(update[top_idx], 0.0) + margins
    ranked[bottom_idx] = torch.clamp_max(update[bottom_idx], 0.0) - margins
    return ranked


def selector_schedule_scale(model_args: ModelArgs, step: int, total_steps: int) -> float:
    schedule = model_args.selector_schedule.lower()
    if schedule == "constant":
        return 1.0
    if schedule == "step":
        if model_args.selector_schedule_step > 0 and step >= model_args.selector_schedule_step:
            return max(0.0, model_args.selector_schedule_scale)
        return 1.0
    if schedule == "linear":
        floor = max(0.0, min(1.0, model_args.selector_schedule_scale))
        end = model_args.selector_schedule_end if model_args.selector_schedule_end > 0 else max(1, total_steps)
        if model_args.selector_schedule_start == 0 or step <= model_args.selector_schedule_start:
            return 1.0
        if step >= end or end <= model_args.selector_schedule_start:
            return floor
        span = max(1, end - model_args.selector_schedule_start)
        progress = (step - model_args.selector_schedule_start) / span
        return 1.0 - progress * (1.0 - floor)
    raise ValueError(f"unsupported selector schedule: {model_args.selector_schedule}")


@dataclass
class BatchSelectorAccumulator:
    signal: List[Optional[torch.Tensor]]
    norm: List[Optional[torch.Tensor]]
    pending_steps: int
    ranked: bool
    selector_update_cutoff: float
    selector_soft_lr: bool
    selector_soft_lr_power: float

    @classmethod
    def new(
        cls,
        model: Transformer,
        ranked: bool,
        selector_update_cutoff: float,
        selector_soft_lr: bool,
        selector_soft_lr_power: float,
    ) -> "BatchSelectorAccumulator":
        signal: List[Optional[torch.Tensor]] = []
        norm: List[Optional[torch.Tensor]] = []
        for layer in model.layers:
            gate = layer.dim_gate
            if gate is None:
                signal.append(None)
                norm.append(None)
                continue
            signal.append(torch.zeros(gate.dim, device=gate.spiker.weights.device, dtype=gate.spiker.weights.dtype))
            norm.append(torch.zeros(gate.dim, device=gate.spiker.weights.device, dtype=gate.spiker.weights.dtype))
        return cls(
            signal=signal,
            norm=norm,
            pending_steps=0,
            ranked=ranked,
            selector_update_cutoff=selector_update_cutoff,
            selector_soft_lr=selector_soft_lr,
            selector_soft_lr_power=selector_soft_lr_power,
        )

    def observe(self, model: Transformer, rewards: torch.Tensor, start_at: int) -> None:
        reward_t = rewards.detach()
        if reward_t.dim() == 1:
            reward_t = reward_t.unsqueeze(0)
        for layer_id, layer in enumerate(model.layers):
            gate = layer.dim_gate
            if gate is None or gate.last_output is None:
                continue
            signal = self.signal[layer_id]
            norm = self.norm[layer_id]
            if signal is None or norm is None:
                continue
            active_bonus = 1.0 + gate.spiker.active_bonus
            gate_input = gate.last_output.gate_input
            selection_mask = gate.last_output.selection_mask
            bsz = min(gate_input.size(0), reward_t.size(0))
            seqlen = min(gate_input.size(1), reward_t.size(1))
            dims = min(signal.numel(), norm.numel(), gate_input.size(2))
            reward_slice = reward_t[:bsz, start_at:seqlen]
            valid_mask = torch.isfinite(reward_slice)
            reward_values = torch.where(valid_mask, reward_slice, torch.zeros_like(reward_slice))
            valid = valid_mask.unsqueeze(-1).to(dtype=signal.dtype)

            contribution = gate_input[:bsz, start_at:seqlen, :dims].abs()
            selected = selection_mask[:bsz, start_at:seqlen, :dims].to(dtype=signal.dtype)

            weighted_contrib = contribution * valid
            signal[:dims] += (reward_values.unsqueeze(-1) * contribution).sum(dim=(0, 1))
            norm[:dims] += weighted_contrib.sum(dim=(0, 1))

            selected_weighted = selected * valid
            signal[:dims] += active_bonus * (reward_values.unsqueeze(-1) * selected).sum(dim=(0, 1))
            norm[:dims] += active_bonus * selected_weighted.sum(dim=(0, 1))
        self.pending_steps += 1

    def apply(self, model: Transformer, selector_lr_scale: float) -> None:
        for layer_id, layer in enumerate(model.layers):
            gate = layer.dim_gate
            signal = self.signal[layer_id]
            norm = self.norm[layer_id]
            if gate is None or signal is None or norm is None:
                continue
            update = torch.zeros_like(signal)
            valid = norm > 1e-6
            update[valid] = signal[valid] / norm[valid]
            if self.ranked:
                update = ranked_batch_signal(update, gate.gate_frac)
            mean_abs = float(update.abs().mean().item())
            if mean_abs > 1e-6:
                update = update / mean_abs
                if self.selector_soft_lr:
                    gate.apply_batch_signal_soft(
                        update,
                        self.selector_update_cutoff,
                        self.selector_soft_lr_power,
                    )
                else:
                    gate.apply_batch_signal_scaled(
                        update,
                        self.selector_update_cutoff,
                        selector_lr_scale,
                    )
            signal.zero_()
            norm.zero_()
        self.pending_steps = 0


def update_gate(
    model: Transformer,
    model_args: ModelArgs,
    token_loss: torch.Tensor,
    step: int,
    total_steps: int,
    batch_accum: Optional[BatchSelectorAccumulator],
) -> None:
    rewards = -token_loss.detach()
    if model_args.center_gate_reward:
        rewards = center_rewards(rewards, start_at=0)

    should_update_gate = model_args.gate_freeze_step == 0 or step <= model_args.gate_freeze_step
    if not should_update_gate:
        return

    if batch_accum is not None:
        batch_accum.observe(model, rewards, start_at=0)
        should_flush = (
            batch_accum.pending_steps >= model_args.batch_selector_steps
            or step == total_steps
            or (model_args.gate_freeze_step > 0 and step == model_args.gate_freeze_step)
        )
        if should_flush:
            batch_accum.apply(model, selector_schedule_scale(model_args, step, total_steps))
        return

    mean_reward = rewards.mean()
    for layer in model.layers:
        if layer.dim_gate is None:
            continue
        layer.dim_gate.apply_token_rewards_from_last(rewards, start_at=0)
        layer.dim_gate.apply_reward(mean_reward)


@torch.inference_mode()
def evaluate_model(
    model: Transformer,
    eval_batches: Sequence[torch.Tensor],
    device: torch.device,
) -> Tuple[float, float, Optional[Dict[int, dict]]]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    snapshot = None
    for batch_id, batch in enumerate(eval_batches):
        set_gate_stats_enabled(model, batch_id == 0)
        inputs, targets = batch_to_inputs_targets(batch, device)
        logits = forward_all_logits(model, inputs)
        loss, _token_loss, accuracy = loss_and_accuracy(logits, targets)
        total_loss += float(loss.item())
        total_acc += accuracy
        if batch_id == 0:
            snapshot = capture_gate_snapshot(model)
    set_gate_stats_enabled(model, True)
    denom = max(1, len(eval_batches))
    return total_loss / denom, total_acc / denom, snapshot


def train_model(
    label: str,
    model_args: ModelArgs,
    train_batches: Sequence[torch.Tensor],
    eval_batches: Sequence[torch.Tensor],
    device: torch.device,
    seed: int,
    lr: float,
    weight_decay: float,
    eval_every: int,
) -> dict:
    torch.manual_seed(seed)
    with torch.device(device):
        model = Transformer(model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    batch_accum = None
    if model_args.use_dim_gate and model_args.batch_selector_steps > 0:
        batch_accum = BatchSelectorAccumulator.new(
            model,
            ranked=model_args.ranked_batch_selector,
            selector_update_cutoff=model_args.selector_update_cutoff,
            selector_soft_lr=model_args.selector_soft_lr,
            selector_soft_lr_power=model_args.selector_soft_lr_power,
        )

    history = []
    previous_snapshot: Optional[Dict[int, dict]] = None
    start_time = time.time()

    for step, batch in enumerate(train_batches, start=1):
        model.train()
        inputs, targets = batch_to_inputs_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_all_logits(model, inputs)
        loss, token_loss, accuracy = loss_and_accuracy(logits, targets)
        loss.backward()
        optimizer.step()

        if model_args.use_dim_gate:
            update_gate(model, model_args, token_loss, step, len(train_batches), batch_accum)

        if step == 1 or step % eval_every == 0 or step == len(train_batches):
            eval_loss, eval_acc, snapshot = evaluate_model(model, eval_batches, device)
            gate_metrics = compare_snapshots(previous_snapshot, snapshot or {})
            previous_snapshot = snapshot
            elapsed = time.time() - start_time
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": accuracy,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "elapsed_s": elapsed,
                "gate_metrics": gate_metrics,
                "gate_snapshot": snapshot,
            }
            history.append(row)
            print(
                "{} step={} train_loss={:.4f} train_acc={:.3f} eval_loss={:.4f} eval_acc={:.3f}".format(
                    label,
                    step,
                    row["train_loss"],
                    row["train_acc"],
                    row["eval_loss"],
                    row["eval_acc"],
                )
            )
            if gate_metrics:
                for layer_id, metrics in sorted(gate_metrics.items()):
                    print(
                        "{} gate layer={} unique_dims={} selection_change_rate={:.3f} mean_selection_jaccard={:.3f}".format(
                            label,
                            layer_id,
                            metrics["unique_dims"],
                            metrics["selection_change_rate"],
                            metrics["mean_selection_jaccard"],
                        )
                    )

    best_eval_acc = max(point["eval_acc"] for point in history)
    final_eval_acc = history[-1]["eval_acc"]
    return {
        "label": label,
        "history": history,
        "best_eval_acc": best_eval_acc,
        "final_eval_acc": final_eval_acc,
    }


def summarize_results(base_result: dict, gated_result: dict) -> None:
    print(
        "summary base_final_eval_acc={:.3f} base_best_eval_acc={:.3f} gated_final_eval_acc={:.3f} gated_best_eval_acc={:.3f}".format(
            base_result["final_eval_acc"],
            base_result["best_eval_acc"],
            gated_result["final_eval_acc"],
            gated_result["best_eval_acc"],
        )
    )
    base_curve = [round(point["eval_acc"], 4) for point in base_result["history"]]
    gated_curve = [round(point["eval_acc"], 4) for point in gated_result["history"]]
    print(f"base_eval_curve={base_curve}")
    print(f"gated_eval_curve={gated_curve}")

    final_gate_metrics = gated_result["history"][-1]["gate_metrics"]
    if final_gate_metrics:
        for layer_id, metrics in sorted(final_gate_metrics.items()):
            print(
                "gated_final_gate layer={} unique_dims={} selection_change_rate={:.3f} mean_selection_jaccard={:.3f} avg_active_dims={:.2f} min_active={} max_active={} mean_abs_weight_delta={:.4f} mean_abs_threshold_delta={:.4f}".format(
                    layer_id,
                    metrics["unique_dims"],
                    metrics["selection_change_rate"],
                    metrics["mean_selection_jaccard"],
                    metrics["stats"]["avg_active_dims"],
                    metrics["stats"]["min_active"],
                    metrics["stats"]["max_active"],
                    metrics["mean_abs_weight_delta"],
                    metrics["mean_abs_threshold_delta"],
                )
            )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config_toy_gate.json"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=6)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--inter-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--kv-lora-rank", type=int, default=32)
    parser.add_argument("--qk-nope-head-dim", type=int, default=16)
    parser.add_argument("--qk-rope-head-dim", type=int, default=8)
    parser.add_argument("--v-head-dim", type=int, default=16)
    parser.add_argument("--gate-stdp-lr", type=float, default=None)
    parser.add_argument("--gate-frac", type=float, default=None)
    parser.add_argument("--gate-softness", type=float, default=None)
    parser.add_argument(
        "--corpus-files",
        nargs="*",
        default=["README.md", "README_DIM_GATE.md", "inference/model.py", "inference/dim_gate.py"],
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else Path(__file__).resolve().parent / args.config
    with open(config_path) as f:
        base_args = ModelArgs(**json.load(f))

    corpus_paths = [repo_root / rel for rel in args.corpus_files]
    corpus = load_corpus(corpus_paths)
    train_data, eval_data = split_corpus(corpus, train_frac=0.9)
    train_batches = make_batches(train_data, args.steps, args.train_batch_size, args.seq_len, args.seed + 1)
    eval_batches = make_batches(eval_data, args.eval_batches, args.eval_batch_size, args.seq_len, args.seed + 2)

    device = select_device(args.device)
    torch.set_default_dtype(default_dtype_for(device, base_args.dtype))

    shared_kwargs = dict(
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        inter_dim=args.inter_dim,
        n_heads=args.n_heads,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
    )
    gate_overrides: Dict[str, object] = {}
    if args.gate_stdp_lr is not None:
        gate_overrides["gate_stdp_lr"] = args.gate_stdp_lr
    if args.gate_frac is not None:
        gate_overrides["gate_frac"] = args.gate_frac
    if args.gate_softness is not None:
        gate_overrides["gate_softness"] = args.gate_softness

    base_model_args = build_model_args(base_args, use_gate=False, gate_overrides=gate_overrides, **shared_kwargs)
    gated_model_args = build_model_args(base_args, use_gate=True, gate_overrides=gate_overrides, **shared_kwargs)

    print(f"device={device}")
    print(f"train_tokens={train_data.numel()} eval_tokens={eval_data.numel()}")
    print(
        "model dim={} layers={} heads={} seq_len={} steps={}".format(
            args.dim,
            args.n_layers,
            args.n_heads,
            args.seq_len,
            args.steps,
        )
    )
    print(
        "gate gate_frac={} gate_softness={} gate_stdp_lr={} batch_selector_steps={} ranked_batch_selector={} selector_schedule={} selector_update_cutoff={} selector_soft_lr={}".format(
            gated_model_args.gate_frac,
            gated_model_args.gate_softness,
            gated_model_args.gate_stdp_lr,
            gated_model_args.batch_selector_steps,
            gated_model_args.ranked_batch_selector,
            gated_model_args.selector_schedule,
            gated_model_args.selector_update_cutoff,
            gated_model_args.selector_soft_lr,
        )
    )

    base_result = train_model(
        label="base",
        model_args=base_model_args,
        train_batches=train_batches,
        eval_batches=eval_batches,
        device=device,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
    )
    gated_result = train_model(
        label="gated",
        model_args=gated_model_args,
        train_batches=train_batches,
        eval_batches=eval_batches,
        device=device,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
    )
    summarize_results(base_result, gated_result)


if __name__ == "__main__":
    main()
