import os
import shutil
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.
        
    Returns:
        None
    """
    assert mp > 0, "Model parallelism (mp) must be greater than 0"
    
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]
    
    hf_ckpt_path = Path(hf_ckpt_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(hf_ckpt_path.glob("*.safetensors")):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue

                param: torch.Tensor = f.get_tensor(name)

                if name.startswith("model."):
                    name = name[len("model."):]

                name = (
                    name.replace("self_attn", "attn")
                    .replace("mlp", "ffn")
                    .replace("weight_scale_inv", "scale")
                    .replace("e_score_correction_bias", "bias")
                )

                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)

                if "experts" in name and "shared_experts" not in name:
                    idx = int(name.split(".")[-3])
                    target_index = idx // n_local_experts
                    if target_index < mp:
                        state_dicts[target_index][name] = param
                elif dim is not None:
                    assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                    shard_size = param.size(dim) // mp
                    for i in range(mp):
                        state_dicts[i][name] = param[:, i * shard_size : (i + 1) * shard_size] if dim == 1 else param[i * shard_size : (i + 1) * shard_size]

    for i in trange(mp):
        save_file(state_dicts[i], save_path / f"model{i}-mp{mp}.safetensors")

    for file_path in hf_ckpt_path.glob("*token*"):
        shutil.copyfile(file_path, save_path / file_path.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()

    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
