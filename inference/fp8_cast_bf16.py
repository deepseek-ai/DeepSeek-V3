import os
import json
from argparse import ArgumentParser
from glob import glob
from typing import Dict
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

# Use the optimized Triton GPU kernel when available; add a CPU fallback below
from kernel import weight_dequant as weight_dequant_gpu

def _infer_device(requested_device: str) -> str:
    """
    Decide which device to use for tensor I/O and dequantization.

    - "auto": prefer CUDA if available, otherwise CPU
    - "cuda": use CUDA when available, else fall back to CPU
    - "cpu": force CPU
    """
    if requested_device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _weight_dequant_cpu(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    CPU fallback for FP8 weight dequantization using pure PyTorch.

    Expands per-block scales to a full resolution map and rescales FP8 weights.
    Prioritizes correctness and portability; may be memory intensive for huge tensors.
    """
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.shape
    m_blocks, n_blocks = s.shape
    assert M % block_size == 0 and N % block_size == 0, "Weight dims must be multiples of block_size"
    assert m_blocks == M // block_size and n_blocks == N // block_size, "Scale shape must match weight tiling"
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    y = x.to(torch.float32) * s_full
    return y.to(torch.get_default_dtype())

def convert_fp8_to_bf16(fp8_path: str, bf16_path: str, device: str = "auto", block_size: int = 128):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.
    device (str): One of {"auto", "cuda", "cpu"}. Controls load/compute device.
    block_size (int): Block size used by quantization/dequantization. Typically 128.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    effective_device = _infer_device(device)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # Cache for loaded safetensor files
    loaded_files: Dict[str, Dict[str, torch.Tensor]] = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device=effective_device)
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device=effective_device)
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    if effective_device == "cuda":
                        new_state_dict[weight_name] = weight_dequant_gpu(weight, scale_inv, block_size=block_size)
                    else:
                        new_state_dict[weight_name] = _weight_dequant_cpu(weight, scale_inv, block_size=block_size)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            if effective_device == "cuda":
                torch.cuda.empty_cache()
    
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Select compute device: 'auto' prefers CUDA if available; otherwise CPU.")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Block size used during quantization/dequantization. Typically 128.")
    args = parser.parse_args()
    convert_fp8_to_bf16(args.input_fp8_hf_path, args.output_bf16_hf_path, device=args.device, block_size=args.block_size)
    
