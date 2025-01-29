import os
import shutil
from argparse import ArgumentParser, ArgumentTypeError
from glob import glob
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


def validate_positive_integer(value):
    """
    Helper function to validate that a value is a positive integer.
    """
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise ArgumentTypeError(f"{value} is not a positive integer")
        return ivalue
    except ValueError:
        raise ArgumentTypeError(f"{value} is not a valid integer")


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
    try:
        torch.set_num_threads(8)
        n_local_experts = n_experts // mp
        state_dicts = [{} for _ in range(mp)]

        if not os.path.exists(hf_ckpt_path):
            raise FileNotFoundError(f"Checkpoint path '{hf_ckpt_path}' does not exist.")

        for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
            if not os.path.isfile(file_path):
                continue
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        if "model.layers.61" in name:
                            continue
                        param: torch.Tensor = f.get_tensor(name)
                        if name.startswith("model."):
                            name = name[len("model.") :]
                        name = name.replace("self_attn", "attn")
                        name = name.replace("mlp", "ffn")
                        name = name.replace("weight_scale_inv", "scale")
                        name = name.replace("e_score_correction_bias", "bias")
                        key = name.split(".")[-2]
                        if key not in mapping:
                            raise KeyError(
                                f"Unexpected key '{key}' in tensor name '{name}'."
                            )
                        new_key, dim = mapping[key]
                        name = name.replace(key, new_key)
                        for i in range(mp):
                            new_param = param
                            if "experts" in name and "shared_experts" not in name:
                                idx = int(name.split(".")[-3])
                                if (
                                    idx < i * n_local_experts
                                    or idx >= (i + 1) * n_local_experts
                                ):
                                    continue
                            elif dim is not None:
                                if param.size(dim) % mp != 0:
                                    raise ValueError(
                                        f"Tensor dimension mismatch for '{name}' (size {param.size(dim)})."
                                    )
                                shard_size = param.size(dim) // mp
                                new_param = param.narrow(
                                    dim, i * shard_size, shard_size
                                ).contiguous()
                            state_dicts[i][name] = new_param
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        os.makedirs(save_path, exist_ok=True)

        for i in trange(mp):
            try:
                save_file(
                    state_dicts[i],
                    os.path.join(save_path, f"model{i}-mp{mp}.safetensors"),
                )
            except Exception as e:
                print(f"Error saving file for model {i}: {e}")
                continue

        for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
            try:
                new_file_path = os.path.join(save_path, os.path.basename(file_path))
                shutil.copyfile(file_path, new_file_path)
            except Exception as e:
                print(f"Error copying token file {file_path}: {e}")
                continue

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hf-ckpt-path", type=str, required=True, help="Path to the checkpoint files."
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Path to save the converted files."
    )
    parser.add_argument(
        "--n-experts",
        type=validate_positive_integer,
        required=True,
        help="Total number of experts in the model.",
    )
    parser.add_argument(
        "--model-parallel",
        type=validate_positive_integer,
        required=True,
        help="Model parallelism factor.",
    )

    args = parser.parse_args()

    if args.n_experts % args.model_parallel != 0:
        raise ValueError(
            f"Number of experts ({args.n_experts}) must be divisible by model parallelism factor ({args.model_parallel})."
        )

    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
