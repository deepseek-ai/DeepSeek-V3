import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
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
    return [{k: v for k, v in row.items() if k != "gate_snapshot"} for row in history]


def curve_metrics(history: List[dict]) -> Dict[str, float]:
    curve = [float(row["eval_acc"]) for row in history]
    if not curve:
        return {"best": 0.0, "final": 0.0, "tail5": 0.0, "mean": 0.0, "drawdown": 0.0}
    best = max(curve)
    final = curve[-1]
    tail = curve[-5:] if len(curve) >= 5 else curve
    best_idx = max(range(len(curve)), key=lambda i: curve[i])
    later = curve[best_idx:]
    return {
        "best": best,
        "final": final,
        "tail5": sum(tail) / len(tail),
        "mean": sum(curve) / len(curve),
        "drawdown": 0.0 if not later else best - min(later),
    }


def mean_curve(results: List[dict]) -> Dict[str, List[float]]:
    histories = [row["history"] for row in results]
    if not histories:
        return {"steps": [], "eval_acc": []}
    steps = [int(point["step"]) for point in histories[0]]
    values = []
    for idx in range(len(histories[0])):
        values.append(sum(float(history[idx]["eval_acc"]) for history in histories) / len(histories))
    return {"steps": steps, "eval_acc": values}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config_toy_gate.json"))
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345, 22345, 32345, 42345, 52345])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=224)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--inter-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--kv-lora-rank", type=int, default=32)
    parser.add_argument("--qk-nope-head-dim", type=int, default=16)
    parser.add_argument("--qk-rope-head-dim", type=int, default=8)
    parser.add_argument("--v-head-dim", type=int, default=16)
    parser.add_argument("--gate-frac", type=float, default=0.09)
    parser.add_argument("--gate-softness", type=float, default=0.25)
    parser.add_argument("--gate-stdp-lr", type=float, default=0.0035)
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

    device = select_device(args.device)
    torch.set_default_dtype(default_dtype_for(device, base_args.dtype))

    corpus_paths = [repo_root / rel for rel in args.corpus_files]
    corpus = load_corpus(corpus_paths)
    train_data, eval_data = split_corpus(corpus, train_frac=0.9)

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
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (repo_root / "results" / f"seed_sweep_native_300steps_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"seeds={args.seeds}")
    print(
        "setup steps={} eval_every={} seq_len={} batch={} lr={} weight_decay={} gate_stdp_lr={} gate_frac={} gate_softness={}".format(
            args.steps,
            args.eval_every,
            args.seq_len,
            args.train_batch_size,
            args.lr,
            args.weight_decay,
            args.gate_stdp_lr,
            args.gate_frac,
            args.gate_softness,
        )
    )
    print(f"output_dir={output_dir}")

    base_rows: List[dict] = []
    gated_rows: List[dict] = []
    per_seed: List[dict] = []

    for seed in args.seeds:
        train_batches = make_batches(train_data, args.steps, args.train_batch_size, args.seq_len, seed + 1)
        eval_batches = make_batches(eval_data, args.eval_batches, args.eval_batch_size, args.seq_len, seed + 2)

        base_model_args = build_model_args(
            base_args,
            use_gate=False,
            gate_overrides={"gate_frac": args.gate_frac, "gate_softness": args.gate_softness},
            **shared_kwargs,
        )
        gated_model_args = build_model_args(
            base_args,
            use_gate=True,
            gate_overrides={
                "gate_frac": args.gate_frac,
                "gate_softness": args.gate_softness,
                "gate_stdp_lr": args.gate_stdp_lr,
                "gate_native_scan": True,
            },
            **shared_kwargs,
        )

        base_result = train_model(
            label=f"base_seed_{seed}",
            model_args=base_model_args,
            train_batches=train_batches,
            eval_batches=eval_batches,
            device=device,
            seed=seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
        )
        gated_result = train_model(
            label=f"gated_seed_{seed}",
            model_args=gated_model_args,
            train_batches=train_batches,
            eval_batches=eval_batches,
            device=device,
            seed=seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
        )

        base_summary = curve_metrics(base_result["history"])
        gated_summary = curve_metrics(gated_result["history"])
        base_rows.append({"seed": seed, "history": sanitize_history(base_result["history"]), **base_summary})
        gated_rows.append({"seed": seed, "history": sanitize_history(gated_result["history"]), **gated_summary})
        row = {
            "seed": seed,
            "base_best": base_summary["best"],
            "base_final": base_summary["final"],
            "base_tail5": base_summary["tail5"],
            "gated_best": gated_summary["best"],
            "gated_final": gated_summary["final"],
            "gated_tail5": gated_summary["tail5"],
            "delta_best": gated_summary["best"] - base_summary["best"],
            "delta_final": gated_summary["final"] - base_summary["final"],
            "delta_tail5": gated_summary["tail5"] - base_summary["tail5"],
            "base_elapsed_s": float(base_result["history"][-1]["elapsed_s"]),
            "gated_elapsed_s": float(gated_result["history"][-1]["elapsed_s"]),
        }
        per_seed.append(row)
        print(
            "seed={} base(best={:.3f}, final={:.3f}, tail5={:.3f}, t={:.1f}s) gated(best={:.3f}, final={:.3f}, tail5={:.3f}, t={:.1f}s) delta(best={:+.3f}, final={:+.3f}, tail5={:+.3f})".format(
                seed,
                row["base_best"],
                row["base_final"],
                row["base_tail5"],
                row["base_elapsed_s"],
                row["gated_best"],
                row["gated_final"],
                row["gated_tail5"],
                row["gated_elapsed_s"],
                row["delta_best"],
                row["delta_final"],
                row["delta_tail5"],
            )
        )

    aggregate = {
        "base_best_mean": sum(row["base_best"] for row in per_seed) / len(per_seed),
        "base_final_mean": sum(row["base_final"] for row in per_seed) / len(per_seed),
        "base_tail5_mean": sum(row["base_tail5"] for row in per_seed) / len(per_seed),
        "gated_best_mean": sum(row["gated_best"] for row in per_seed) / len(per_seed),
        "gated_final_mean": sum(row["gated_final"] for row in per_seed) / len(per_seed),
        "gated_tail5_mean": sum(row["gated_tail5"] for row in per_seed) / len(per_seed),
        "delta_best_mean": sum(row["delta_best"] for row in per_seed) / len(per_seed),
        "delta_final_mean": sum(row["delta_final"] for row in per_seed) / len(per_seed),
        "delta_tail5_mean": sum(row["delta_tail5"] for row in per_seed) / len(per_seed),
        "base_elapsed_mean_s": sum(row["base_elapsed_s"] for row in per_seed) / len(per_seed),
        "gated_elapsed_mean_s": sum(row["gated_elapsed_s"] for row in per_seed) / len(per_seed),
    }
    print("aggregate", aggregate)

    base_curve = mean_curve(base_rows)
    gated_curve = mean_curve(gated_rows)
    plt.figure(figsize=(9, 5))
    plt.plot(base_curve["steps"], base_curve["eval_acc"], marker="o", label="base mean")
    plt.plot(gated_curve["steps"], gated_curve["eval_acc"], marker="o", label="gated mean")
    plt.xlabel("Step")
    plt.ylabel("Eval Accuracy")
    plt.title("Seed Sweep Mean Eval Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_eval_accuracy.png", dpi=150)
    plt.close()

    with open(output_dir / "seed_sweep_results.json", "w") as f:
        json.dump(
            {
                "config": {
                    "seeds": args.seeds,
                    "steps": args.steps,
                    "eval_every": args.eval_every,
                    "seq_len": args.seq_len,
                    "train_batch_size": args.train_batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "eval_batches": args.eval_batches,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "gate_stdp_lr": args.gate_stdp_lr,
                    "gate_frac": args.gate_frac,
                    "gate_softness": args.gate_softness,
                    "device": str(device),
                },
                "per_seed": per_seed,
                "aggregate": aggregate,
                "base_results": base_rows,
                "gated_results": gated_rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
