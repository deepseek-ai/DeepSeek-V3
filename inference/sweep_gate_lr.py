import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import torch

from model import ModelArgs
from train_gate_compare import (
    build_model_args,
    default_dtype_for,
    load_corpus,
    make_batches,
    select_device,
    split_corpus,
    train_model,
)


def sanitize_history(history: List[dict]) -> List[dict]:
    sanitized = []
    for row in history:
        clean = {k: v for k, v in row.items() if k != "gate_snapshot"}
        sanitized.append(clean)
    return sanitized


def curve_metrics(history: List[dict]) -> Dict[str, float]:
    curve = [float(row["eval_acc"]) for row in history]
    if not curve:
        return {
            "mean_eval_acc": 0.0,
            "tail4_mean": 0.0,
            "tail5_mean": 0.0,
            "tail6_mean": 0.0,
            "post_peak_drawdown": 0.0,
            "best_step": 0.0,
        }
    best_idx = max(range(len(curve)), key=lambda i: curve[i])
    best = curve[best_idx]
    later = curve[best_idx:]
    post_peak_drawdown = 0.0 if not later else best - min(later)
    tail4 = curve[-4:] if len(curve) >= 4 else curve
    tail5 = curve[-5:] if len(curve) >= 5 else curve
    tail6 = curve[-6:] if len(curve) >= 6 else curve
    return {
        "mean_eval_acc": sum(curve) / len(curve),
        "tail4_mean": sum(tail4) / len(tail4),
        "tail5_mean": sum(tail5) / len(tail5),
        "tail6_mean": sum(tail6) / len(tail6),
        "post_peak_drawdown": post_peak_drawdown,
        "best_step": float(history[best_idx]["step"]),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config_toy_gate.json"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=224)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--inter-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--kv-lora-rank", type=int, default=32)
    parser.add_argument("--qk-nope-head-dim", type=int, default=8)
    parser.add_argument("--qk-rope-head-dim", type=int, default=8)
    parser.add_argument("--v-head-dim", type=int, default=16)
    parser.add_argument("--gate-frac", type=float, default=0.09)
    parser.add_argument("--gate-softness", type=float, default=0.25)
    parser.add_argument("--gate-lrs", nargs="+", type=float, default=[0.0015, 0.0025, 0.0035, 0.005, 0.0075])
    parser.add_argument("--output-dir", type=Path, default=None)
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

    base_gate_overrides: Dict[str, object] = {
        "gate_frac": args.gate_frac,
        "gate_softness": args.gate_softness,
    }
    base_model_args = build_model_args(base_args, use_gate=False, gate_overrides=base_gate_overrides, **shared_kwargs)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (repo_root / "results" / f"gate_lr_sweep_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"train_tokens={train_data.numel()} eval_tokens={eval_data.numel()}")
    print(
        "match target lr={} weight_decay={} seed={} gate_frac={} gate_softness={} batch_selector_steps={} ranked_batch_selector={} selector_schedule={} selector_update_cutoff={} selector_soft_lr={}".format(
            args.lr,
            args.weight_decay,
            args.seed,
            args.gate_frac,
            args.gate_softness,
            base_args.batch_selector_steps,
            base_args.ranked_batch_selector,
            base_args.selector_schedule,
            base_args.selector_update_cutoff,
            base_args.selector_soft_lr,
        )
    )
    print(
        "toy model dim={} layers={} heads={} seq_len={} steps={} batch={}".format(
            args.dim,
            args.n_layers,
            args.n_heads,
            args.seq_len,
            args.steps,
            args.train_batch_size,
        )
    )
    print(f"output_dir={output_dir}")

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

    summaries: List[dict] = []
    results_by_lr: Dict[str, dict] = {}
    base_curve_summary = curve_metrics(base_result["history"])
    print(
        "base_reference best_eval_acc={:.3f} final_eval_acc={:.3f}".format(
            base_result["best_eval_acc"],
            base_result["final_eval_acc"],
        )
    )
    for gate_lr in args.gate_lrs:
        gate_overrides = {
            "gate_frac": args.gate_frac,
            "gate_softness": args.gate_softness,
            "gate_stdp_lr": gate_lr,
        }
        gated_model_args = build_model_args(base_args, use_gate=True, gate_overrides=gate_overrides, **shared_kwargs)
        result = train_model(
            label=f"gated_lr_{gate_lr}",
            model_args=gated_model_args,
            train_batches=train_batches,
            eval_batches=eval_batches,
            device=device,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
        )
        final_metrics = result["history"][-1]["gate_metrics"]
        layer0 = final_metrics.get(0, {})
        summary = curve_metrics(result["history"])
        summaries.append(
            {
                "gate_lr": gate_lr,
                "best_eval_acc": result["best_eval_acc"],
                "final_eval_acc": result["final_eval_acc"],
                "mean_eval_acc": summary["mean_eval_acc"],
                "tail4_mean": summary["tail4_mean"],
                "tail5_mean": summary["tail5_mean"],
                "tail6_mean": summary["tail6_mean"],
                "post_peak_drawdown": summary["post_peak_drawdown"],
                "best_step": summary["best_step"],
                "delta_best_vs_base": result["best_eval_acc"] - base_result["best_eval_acc"],
                "delta_final_vs_base": result["final_eval_acc"] - base_result["final_eval_acc"],
                "delta_mean_vs_base": summary["mean_eval_acc"] - base_curve_summary["mean_eval_acc"],
                "delta_tail5_vs_base": summary["tail5_mean"] - base_curve_summary["tail5_mean"],
                "layer0_change_rate": layer0.get("selection_change_rate", 0.0),
                "layer0_jaccard": layer0.get("mean_selection_jaccard", 1.0),
                "layer0_avg_active": layer0.get("stats", {}).get("avg_active_dims", 0.0),
                "layer0_weight_delta": layer0.get("mean_abs_weight_delta", 0.0),
                "layer0_threshold_delta": layer0.get("mean_abs_threshold_delta", 0.0),
                "elapsed_s": result["history"][-1]["elapsed_s"],
            }
        )
        results_by_lr[str(gate_lr)] = {
            "best_eval_acc": result["best_eval_acc"],
            "final_eval_acc": result["final_eval_acc"],
            "history": sanitize_history(result["history"]),
        }
        print(
            "sweep gate_lr={} best_eval_acc={:.3f} final_eval_acc={:.3f} tail5_mean={:.3f} drawdown={:.3f} delta_best_vs_base={:+.3f} delta_final_vs_base={:+.3f} delta_tail5_vs_base={:+.3f} layer0_change_rate={:.3f} layer0_jaccard={:.3f} layer0_avg_active={:.2f} layer0_weight_delta={:.4f} layer0_threshold_delta={:.4f}".format(
                gate_lr,
                result["best_eval_acc"],
                result["final_eval_acc"],
                summary["tail5_mean"],
                summary["post_peak_drawdown"],
                result["best_eval_acc"] - base_result["best_eval_acc"],
                result["final_eval_acc"] - base_result["final_eval_acc"],
                summary["tail5_mean"] - base_curve_summary["tail5_mean"],
                layer0.get("selection_change_rate", 0.0),
                layer0.get("mean_selection_jaccard", 1.0),
                layer0.get("stats", {}).get("avg_active_dims", 0.0),
                layer0.get("mean_abs_weight_delta", 0.0),
                layer0.get("mean_abs_threshold_delta", 0.0),
            )
        )

    ranked = sorted(summaries, key=lambda row: (row["best_eval_acc"], row["final_eval_acc"]), reverse=True)
    ranked_tail = sorted(summaries, key=lambda row: (row["tail5_mean"], row["final_eval_acc"]), reverse=True)
    ranked_drawdown = sorted(summaries, key=lambda row: (row["post_peak_drawdown"], -row["final_eval_acc"]))
    print("sweep_ranked")
    for row in ranked:
        print(
            "gate_lr={} best_eval_acc={:.3f} final_eval_acc={:.3f} tail5_mean={:.3f} drawdown={:.3f} delta_best_vs_base={:+.3f} delta_final_vs_base={:+.3f}".format(
                row["gate_lr"],
                row["best_eval_acc"],
                row["final_eval_acc"],
                row["tail5_mean"],
                row["post_peak_drawdown"],
                row["delta_best_vs_base"],
                row["delta_final_vs_base"],
            )
        )
    print("sweep_ranked_tail")
    for row in ranked_tail:
        print(
            "gate_lr={} tail5_mean={:.3f} final_eval_acc={:.3f} drawdown={:.3f} delta_tail5_vs_base={:+.3f}".format(
                row["gate_lr"],
                row["tail5_mean"],
                row["final_eval_acc"],
                row["post_peak_drawdown"],
                row["delta_tail5_vs_base"],
            )
        )
    print("sweep_ranked_drawdown")
    for row in ranked_drawdown:
        print(
            "gate_lr={} drawdown={:.3f} tail5_mean={:.3f} final_eval_acc={:.3f}".format(
                row["gate_lr"],
                row["post_peak_drawdown"],
                row["tail5_mean"],
                row["final_eval_acc"],
            )
        )

    payload = {
        "config": {
            "device": str(device),
            "seed": args.seed,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "eval_batches": args.eval_batches,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dim": args.dim,
            "n_layers": args.n_layers,
            "inter_dim": args.inter_dim,
            "n_heads": args.n_heads,
            "kv_lora_rank": args.kv_lora_rank,
            "qk_nope_head_dim": args.qk_nope_head_dim,
            "qk_rope_head_dim": args.qk_rope_head_dim,
            "v_head_dim": args.v_head_dim,
            "gate_frac": args.gate_frac,
            "gate_softness": args.gate_softness,
            "gate_lrs": args.gate_lrs,
            "batch_selector_steps": base_args.batch_selector_steps,
            "ranked_batch_selector": base_args.ranked_batch_selector,
            "selector_schedule": base_args.selector_schedule,
            "selector_update_cutoff": base_args.selector_update_cutoff,
            "selector_soft_lr": base_args.selector_soft_lr,
        },
        "base_result": {
            "best_eval_acc": base_result["best_eval_acc"],
            "final_eval_acc": base_result["final_eval_acc"],
            "history": sanitize_history(base_result["history"]),
        },
        "base_curve_summary": base_curve_summary,
        "sweep_summaries": summaries,
        "ranked": ranked,
        "ranked_tail": ranked_tail,
        "ranked_drawdown": ranked_drawdown,
    }
    payload["gated_results"] = results_by_lr

    json_path = output_dir / "sweep_results.json"
    json_path.write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        base_steps = [row["step"] for row in payload["base_result"]["history"]]
        base_curve = [row["eval_acc"] for row in payload["base_result"]["history"]]
        ax.plot(base_steps, base_curve, marker="o", linewidth=2.5, label="base", color="#1f2937")
        palette = ["#2563eb", "#0f766e", "#c2410c", "#7c3aed", "#be123c", "#0891b2"]
        for idx, gate_lr in enumerate(args.gate_lrs):
            result = payload["gated_results"][str(gate_lr)]
            steps = [row["step"] for row in result["history"]]
            curve = [row["eval_acc"] for row in result["history"]]
            ax.plot(
                steps,
                curve,
                marker="o",
                linewidth=2.0,
                label=f"gated lr={gate_lr:g}",
                color=palette[idx % len(palette)],
            )
        ax.set_title("Base vs Gated Eval Accuracy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Eval Accuracy")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        curve_path = output_dir / "eval_accuracy_curves.png"
        fig.savefig(curve_path, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [f"{row['gate_lr']:g}" for row in ranked]
        values = [row["best_eval_acc"] for row in ranked]
        colors = ["#2563eb" if i == 0 else "#94a3b8" for i in range(len(labels))]
        ax.bar(labels, values, color=colors)
        ax.axhline(base_result["best_eval_acc"], color="#111827", linestyle="--", linewidth=1.5, label="base best")
        ax.set_title("Gate LR Sweep Ranked by Best Eval Accuracy")
        ax.set_xlabel("gate_stdp_lr")
        ax.set_ylabel("Best Eval Accuracy")
        ax.legend()
        fig.tight_layout()
        rank_path = output_dir / "sweep_ranking.png"
        fig.savefig(rank_path, dpi=160)
        plt.close(fig)
        print(f"wrote_json={json_path}")
        print(f"wrote_plot={curve_path}")
        print(f"wrote_plot={rank_path}")
    except Exception as exc:
        print(f"plot_error={exc!r}")


if __name__ == "__main__":
    main()
