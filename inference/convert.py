import os
import shutil
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm, trange

# Constants and type definitions
TensorMapping = Dict[str, Tuple[str, Optional[int]]]
StateDict = Dict[str, torch.Tensor]

# Define mapping as a constant at module level
TENSOR_MAPPING: TensorMapping = {
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

def process_tensor_name(name: str) -> str:
    """
    Process tensor name by removing prefixes and replacing common patterns.
    
    Args:
        name: Original tensor name
        
    Returns:
        Processed tensor name
    """
    if name.startswith("model."):
        name = name[len("model."):]
    
    replacements = {
        "self_attn": "attn",
        "mlp": "ffn",
        "weight_scale_inv": "scale",
        "e_score_correction_bias": "bias"
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name

def shard_tensor(param: torch.Tensor, mp_idx: int, mp_count: int, dim: int) -> torch.Tensor:
    """
    Shard a tensor along specified dimension for model parallelism.
    
    Args:
        param: Input tensor to shard
        mp_idx: Index of current model parallel rank
        mp_count: Total number of model parallel ranks
        dim: Dimension along which to shard
        
    Returns:
        Sharded tensor slice
    """
    if param.size(dim) % mp_count != 0:
        raise ValueError(f"Tensor size {param.size(dim)} not divisible by mp_count {mp_count}")
    
    shard_size = param.size(dim) // mp_count
    return param.narrow(dim, mp_idx * shard_size, shard_size).contiguous()

def convert_checkpoint(
    hf_ckpt_path: Union[str, Path],
    save_path: Union[str, Path],
    n_experts: int,
    mp: int
) -> None:
    """
    Convert and save model checkpoint files into a specified format.
    
    Args:
        hf_ckpt_path: Path to input checkpoint directory
        save_path: Path to output directory for converted checkpoints
        n_experts: Total number of experts in model
        mp: Model parallelism factor
        
    Raises:
        ValueError: If n_experts is not divisible by mp
        FileNotFoundError: If input path doesn't exist or contain safetensors
    """
    if n_experts % mp != 0:
        raise ValueError(f"Number of experts ({n_experts}) must be divisible by model parallel size ({mp})")
    
    hf_ckpt_path = Path(hf_ckpt_path)
    save_path = Path(save_path)
    
    if not hf_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path {hf_ckpt_path} does not exist")
        
    safetensor_files = list(hf_ckpt_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {hf_ckpt_path}")

    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts: List[StateDict] = [{} for _ in range(mp)]

    # Process each checkpoint file
    for file_path in tqdm(safetensor_files, desc="Processing checkpoint files"):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                    
                param: torch.Tensor = f.get_tensor(name)
                name = process_tensor_name(name)
                
                key = name.split(".")[-2]
                if key not in TENSOR_MAPPING:
                    raise ValueError(f"Unknown tensor key: {key}")
                    
                new_key, dim = TENSOR_MAPPING[key]
                name = name.replace(key, new_key)
                
                # Distribute tensors across model parallel ranks
                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if not (i * n_local_experts <= idx < (i + 1) * n_local_experts):
                            continue
                    elif dim is not None:
                        new_param = shard_tensor(param, i, mp, dim)
                    state_dicts[i][name] = new_param

    # Save converted checkpoints
    save_path.mkdir(parents=True, exist_ok=True)
    
    for i in trange(mp, desc="Saving converted checkpoints"):
        output_file = save_path / f"model{i}-mp{mp}.safetensors"
        save_file(state_dicts[i], str(output_file))

    # Copy tokenizer files
    for file_path in hf_ckpt_path.glob("*token*"):
        shutil.copyfile(file_path, save_path / file_path.name)

def main():
    """Parse command line arguments and run the conversion."""
    parser = ArgumentParser(description="Convert HuggingFace checkpoints to custom format")
    parser.add_argument("--hf-ckpt-path", type=str, required=True,
                      help="Path to input HuggingFace checkpoint directory")
    parser.add_argument("--save-path", type=str, required=True,
                      help="Path to output directory for converted checkpoints")
    parser.add_argument("--n-experts", type=int, required=True,
                      help="Total number of experts in the model")
    parser.add_argument("--model-parallel", type=int, required=True,
                      help="Model parallelism factor")
    
    args = parser.parse_args()
    
    try:
        convert_checkpoint(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()