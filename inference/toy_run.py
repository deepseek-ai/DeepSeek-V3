import json
from argparse import ArgumentParser

import torch

from model import ModelArgs, Transformer


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
        next_tokens = logits.argmax(dim=-1, keepdim=True)
        next_logits = model(next_tokens, start_pos=prompt_len)

    print(f"device={device}")
    print(f"prompt_tokens={tuple(tokens.shape)}")
    print(f"logits={tuple(logits.shape)}")
    print(f"next_logits={tuple(next_logits.shape)}")

    for layer_id, layer in enumerate(model.layers):
        if layer.dim_gate is None:
            continue
        stats = layer.dim_gate.last_stats.summary()
        print(
            "layer={} gate tokens={} avg_active_dims={:.2f} min_active={} max_active={} "
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


if __name__ == "__main__":
    main()
