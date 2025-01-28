import os
import shutil
import logging
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from tqdm import tqdm
import torch
from safetensors.torch import safe_open, save_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type aliases
TensorMapping = Dict[str, Tuple[str, Optional[int]]]

# Configuration mapping with type hints
MAPPING: TensorMapping = {
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

def validate_paths(hf_ckpt_path: str, save_path: str) -> None:
    """Validate input and output paths."""
    if not os.path.isdir(hf_ckpt_path):
        logger.error(f"Input directory {hf_ckpt_path} does not exist")
        raise ValueError(f"Input directory {hf_ckpt_path} does not exist")
    
    os.makedirs(save_path, exist_ok=True)
    if not os.access(save_path, os.W_OK):
        logger.error(f"No write permission for output directory {save_path}")
        raise PermissionError(f"No write permission for output directory {save_path}")

def process_tensor_name(name: str) -> str:
    """Process and normalize tensor names."""
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

def split_tensor(param: torch.Tensor, dim: Optional[int], mp: int, idx: int) -> torch.Tensor:
    """Split tensor for model parallelism."""
    if dim is None:
        return param
    
    if param.size(dim) % mp != 0:
        logger.error(f"Dimension {dim} of tensor with shape {param.shape} is not divisible by model parallelism factor {mp}")
        raise ValueError(f"Dimension {dim} of tensor with shape {param.shape} is not divisible by model parallelism factor {mp}")
    
    shard_size = param.size(dim) // mp
    return param.narrow(dim, idx * shard_size, shard_size).contiguous()

def process_checkpoint_files(
    hf_ckpt_path: str,
    mp: int,
    n_local_experts: int,
    state_dicts: List[Dict[str, torch.Tensor]]
) -> None:
    """Process all checkpoint files and populate state dictionaries."""
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors")), desc="Processing checkpoint files"):
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in tqdm(f.keys(), desc=f"Processing {os.path.basename(file_path)}", leave=False):
                    if "model.layers.61" in name:
                        logger.debug(f"Skipping layer 61 tensor: {name}")
                        continue
                    
                    param = f.get_tensor(name)
                    processed_name = process_tensor_name(name)
                    
                    key = processed_name.split(".")[-2]
                    if key not in MAPPING:
                        logger.error(f"Unexpected tensor key: {key} in tensor {name}")
                        raise KeyError(f"Unexpected tensor key: {key} in tensor {name}")
                    
                    new_key, dim = MAPPING[key]
                    final_name = processed_name.replace(key, new_key)
                    
                    for i in range(mp):
                        if "experts" in final_name and "shared_experts" not in final_name:
                            expert_idx = int(final_name.split(".")[-3])
                            if not (i * n_local_experts <= expert_idx < (i + 1) * n_local_experts):
                                continue
                        
                        split_param = split_tensor(param, dim, mp, i)
                        state_dicts[i][final_name] = split_param
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

def save_output_files(
    state_dicts: List[Dict[str, torch.Tensor]],
    save_path: str,
    mp: int,
    hf_ckpt_path: str
) -> None:
    """Save processed state dictionaries and copy token files."""
    for i in tqdm(range(mp), desc="Saving output files"):
        output_file = os.path.join(save_path, f"model{i}-mp{mp}.safetensors")
        save_file(state_dicts[i], output_file, metadata={"format": "pt"})
    
    copy_token_files(hf_ckpt_path, save_path)

def copy_token_files(hf_ckpt_path: str, save_path: str) -> None:
    """Copy token-related files from the checkpoint path to the save path."""
    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        try:
            shutil.copy(file_path, os.path.join(save_path, os.path.basename(file_path)))
        except IOError as e:
            logger.error(f"Error copying file {file_path}: {str(e)}")

def main(
    hf_ckpt_path: str,
    save_path: str,
    n_experts: int,
    mp: int
) -> None:
    """
    Convert and split model checkpoints for distributed training.
    
    Args:
        hf_ckpt_path: Path to HuggingFace format checkpoint directory
        save_path: Output directory for converted checkpoints
        n_experts: Total number of experts in the model
        mp: Model parallelism factor
    """
    torch.set_num_threads(8)
    validate_paths(hf_ckpt_path, save_path)
    
    if n_experts % mp != 0:
        raise ValueError(f"Number of experts {n_experts} must be divisible by model parallelism factor {mp}")
    
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]
    
    process_checkpoint_files(hf_ckpt_path, mp, n_local_experts, state_dicts)
    save_output_files(state_dicts, save_path, mp, hf_ckpt_path)
    
    logger.info(f"Successfully converted checkpoints. Output saved to {save_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert HuggingFace checkpoints to distributed format")
    parser.add_argument(
        "--hf-ckpt-path",
        type=str,
        required=True,
        help="Path to input HuggingFace checkpoint directory"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Output directory for converted checkpoints"
    )
    parser.add_argument(
        "--n-experts",
        type=int,
        required=True,
        help="Total number of experts in the model"
    )
    parser.add_argument(
        "--model-parallel",
        type=int,
        required=True,
        dest="model_parallel",
        help="Model parallelism factor"
    )
    
    args = parser.parse_args()
    
    try:
        main(
            args.hf_ckpt_path,
            args.save_path,
            args.n_experts,
            args.model_parallel
        )
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise
