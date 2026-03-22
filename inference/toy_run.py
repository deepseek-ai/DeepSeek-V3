import json
from argparse import ArgumentParser

import torch

from model import ModelArgs, Transformer


def collect_gate_stats(model: Transformer):
    stats = []
    for layer_id, layer in enumerate(model.layers):
        if layer.dim_gate is None:
            continue
        stats.append((layer_id, layer.dim_gate.last_stats.summary()))
    return stats


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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_toy_gate.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--simulate-learning", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))

    device = select_device(args.device)
    torch.set_default_dtype(default_dtype_for(device, model_args.dtype))
    torch.manual_seed(args.seed)

    batch_size = min(args.batch_size, model_args.max_batch_size)
    prompt_len = min(args.prompt_len, model_args.max_seq_len - 1)

    with torch.device(device):
        model = Transformer(model_args)
    model.eval()

    tokens = torch.randint(
        0,
        model_args.vocab_size,
        (batch_size, prompt_len),
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits = model(tokens, start_pos=0)
        prompt_gate_stats = collect_gate_stats(model)
        learning_stats = []
        if args.simulate_learning:
            rewards = torch.linspace(-0.5, 0.5, prompt_len, device=device, dtype=torch.float32)
            rewards = rewards.unsqueeze(0).expand(batch_size, prompt_len)
            for layer_id, layer in enumerate(model.layers):
                if layer.dim_gate is None:
                    continue
                before = layer.dim_gate.spiker.weights.detach().clone()
                layer.dim_gate.apply_token_rewards_from_last(rewards, start_at=max(0, prompt_len // 4))
                after_token = layer.dim_gate.spiker.weights.detach().clone()
                signal = torch.linspace(-1.0, 1.0, layer.dim_gate.dim, device=device, dtype=torch.float32)
                layer.dim_gate.apply_batch_signal_scaled(signal, cutoff=0.2, scale=0.5)
                after_batch = layer.dim_gate.spiker.weights.detach()
                token_delta = (after_token - before).abs()
                batch_delta = (after_batch - after_token).abs()
                learning_stats.append(
                    (
                        layer_id,
                        float(token_delta.mean().item()),
                        float(token_delta.max().item()),
                        float(batch_delta.mean().item()),
                        float(batch_delta.max().item()),
                        float(layer.dim_gate.spiker.reward_ema.item()),
                    )
                )
        next_tokens = logits.argmax(dim=-1, keepdim=True)
        next_logits = model(next_tokens, start_pos=prompt_len)
        decode_gate_stats = collect_gate_stats(model)

    print(f"device={device}")
    print(f"prompt_tokens={tuple(tokens.shape)}")
    print(f"logits={tuple(logits.shape)}")
    print(f"next_logits={tuple(next_logits.shape)}")

    for label, gate_stats in (("prompt", prompt_gate_stats), ("decode", decode_gate_stats)):
        print(f"{label}_gate_stats:")
        for layer_id, stats in gate_stats:
            print(
                "  layer={} tokens={} avg_active_dims={:.2f} min_active={} max_active={} "
                "avg_raw_active_dims={:.2f} clamped_min_tokens={} clamped_max_tokens={}".format(
                    layer_id,
                    stats["tokens"],
                    stats["avg_active_dims"],
                    stats["min_active"],
                    stats["max_active"],
                    stats["avg_raw_active_dims"],
                    stats["clamped_min_tokens"],
                    stats["clamped_max_tokens"],
                )
            )

    for layer_id, token_mean_delta, token_max_delta, batch_mean_delta, batch_max_delta, reward_ema in learning_stats:
        print(
            "layer={} learning token_reward_mean_abs_weight_delta={:.6f} token_reward_max_abs_weight_delta={:.6f} "
            "batch_signal_mean_abs_weight_delta={:.6f} batch_signal_max_abs_weight_delta={:.6f} reward_ema={:.6f}".format(
                layer_id,
                token_mean_delta,
                token_max_delta,
                batch_mean_delta,
                batch_max_delta,
                reward_ema,
            )
        )


if __name__ == "__main__":
    main()
