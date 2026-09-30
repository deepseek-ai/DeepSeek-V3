import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file, safe_open

from kernel import weight_dequant

def _validate_fp8_scales(safetensor_files, weight_map, block_size=128):
    """Check FP8 scale dependencies using only safetensors headers on CPU.

    Resolve scales through the index, not through the weight's own shard:
    a weight and its scale may legitimately live in different files.
    """
    fp8_weights = []
    scales = {}
    for file_path in safetensor_files:
        file_name = os.path.basename(file_path)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_slice(name)
                if name.endswith("_scale_inv"):
                    scales[(file_name, name)] = tensor.get_shape()
                elif tensor.get_dtype() in ("F8_E4M3", "F8_E5M2"):
                    fp8_weights.append((name, tensor.get_shape()))

    for name, shape in fp8_weights:
        scale_name = f"{name}_scale_inv"
        scale_file = weight_map.get(scale_name)
        scale_shape = scales.get((scale_file, scale_name))
        if scale_shape is None:
            raise KeyError(
                f"Missing scale_inv tensor for {name}: {scale_name} "
                f"is not available in its indexed shard {scale_file!r}"
            )
        if len(shape) != 2:
            raise ValueError(f"FP8 weight {name} must be 2-D, got shape {shape}")
        expected_shape = [(size + block_size - 1) // block_size for size in shape]
        if scale_shape != expected_shape:
            raise ValueError(
                f"Invalid scale shape for {name}: {scale_name} has shape "
                f"{scale_shape}, expected {expected_shape}"
            )


def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.
    ValueError: If an FP8 weight or its scale has an unsupported shape.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - FP8 scale dependencies are validated before loading tensors onto the GPU
      or writing output files. Missing or malformed scales fail preflight.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    safetensor_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
    _validate_fp8_scales(safetensor_files, weight_map)
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    
    # Cache for loaded safetensor files
    loaded_files = {}
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
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                scale_inv_name = f"{weight_name}_scale_inv"
                # A missing scale must fail, never produce a mixed FP8/BF16 output.
                scale_inv = get_tensor(scale_inv_name)
                new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                fp8_weight_names.append(weight_name)
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
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
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    
