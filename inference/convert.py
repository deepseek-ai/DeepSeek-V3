import os
import shutil
import mmap
import threading
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from safetensors.torch import safe_open, save_file
from collections import defaultdict

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

# Thread Lock for Safe Dictionary Access
state_lock = threading.Lock()

def fast_copy(src: Path, dst: Path):
    """Efficiently copies large files using shutil for optimal memory usage"""
    if dst.exists():
        dst.unlink()  # Remove file if it already exists
    if src.stat().st_size < 10 * 1024 * 1024:  # If file < 10MB, use shutil
        shutil.copyfile(src, dst)
    else:
        with open(src, "rb") as f_src, open(dst, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst, length=16*1024*1024)

def copy_token_file(file_path, save_path):
    """Helper function for parallel copying of token files"""
    fast_copy(file_path, Path(save_path) / file_path.name)

def inner_safe_open(name: str, f, mp, state_dicts, n_local_experts): 
    """Processes tensor files and maps keys correctly"""
    with torch.no_grad():
        param: torch.Tensor = f.get_tensor(name)
        name = name[len("model."):] if name.startswith("model.") else name
        name = name.replace("self_attn", "attn").replace("mlp", "ffn")
        name = name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")
        key = name.split(".")[-2]
        assert key in mapping
        new_key, dim = mapping[key]
        name = name.replace(key, new_key)

        for i in range(mp):
            new_param = param
            if "experts" in name and "shared_experts" not in name:
                idx = int(name.split(".")[-3])
                if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                    continue
            elif dim is not None:
                shard_size = param.size(dim) // mp
                new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()

            # Lock to avoid race conditions
            with state_lock:
                state_dicts[i][name] = new_param

def process_file(file_path, mp, state_dicts, n_local_experts):
    """Processes a single safetensor file"""
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            if "model.layers.61" not in name:
                inner_safe_open(name, f, mp, state_dicts, n_local_experts)

def main(hf_ckpt_path, save_path, n_experts, mp):
    """Converts and saves model checkpoint files into a specified format."""
    n_local_experts = n_experts // mp

    # Use defaultdict to prevent key errors in multi-threading
    state_dicts = [defaultdict(dict) for _ in range(mp)]
    
    file_list = list(Path(hf_ckpt_path).glob("*.safetensors"))
    token_files = list(Path(hf_ckpt_path).glob("*token*"))

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Parallel Processing with ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, file, mp, state_dicts, n_local_experts): file
            for file in file_list
        }
        for future in tqdm(as_completed(futures), desc="Processing safetensors", total=len(file_list)):
            future.result()  # Ensure exceptions are raised

    # Save processed model shards
    for i in trange(mp, desc="Saving model shards"):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    # Parallel Token File Copying
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(copy_token_file, file, save_path): file
            for file in token_files
        }
        for future in tqdm(as_completed(futures), desc="Copying token files", total=len(token_files)):
            future.result()  # Ensure exceptions are raised

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()

    assert args.n_experts % args.model_parallel == 0, "n_experts must be divisible by model_parallel"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
