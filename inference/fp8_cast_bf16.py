import os
import json
import shutil
import tempfile
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file, safe_open

from kernel import weight_dequant


def _remove_path(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.exists(path):
        shutil.rmtree(path)


def _publish_staged_directory(staging_path, destination):
    """Replace destination with a completed sibling directory, rolling back on error."""
    backup_path = None
    if os.path.lexists(destination):
        backup_path = tempfile.mkdtemp(
            prefix=f".{os.path.basename(destination)}.backup-",
            dir=os.path.dirname(destination),
        )
        os.rmdir(backup_path)
        try:
            os.replace(destination, backup_path)
        except BaseException:
            backup_path = None
            raise

    try:
        os.replace(staging_path, destination)
    except BaseException:
        if backup_path is not None:
            os.replace(backup_path, destination)
            backup_path = None
        raise

    if backup_path is not None:
        # Publication succeeded. A failed cleanup must not report a failed
        # conversion after the new output has already become visible.
        try:
            _remove_path(backup_path)
        except OSError:
            pass


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
    ValueError: If the output would contain the input or staged output is invalid.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    - Shards and the index are staged and validated before the output is published.
    """
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    destination = os.path.abspath(bf16_path)
    source = os.path.abspath(fp8_path)
    if destination == source or os.path.commonpath((source, destination)) == destination:
        raise ValueError("The output directory must not be the input directory or contain it")
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    staging_path = tempfile.mkdtemp(
        prefix=f".{os.path.basename(destination)}.staging-", dir=parent
    )
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        _convert_to_directory(fp8_path, weight_map, staging_path)
        _validate_staged_output(staging_path)
        _publish_staged_directory(staging_path, destination)
    except BaseException:
        if os.path.lexists(staging_path):
            _remove_path(staging_path)
        raise
    finally:
        torch.set_default_dtype(old_dtype)


def _convert_to_directory(fp8_path, weight_map, output_path):
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

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                # Any missing scale must fail the staged conversion. Publishing
                # the original FP8 tensor would make the output inconsistent.
                scale_inv = get_tensor(scale_inv_name)
                fp8_weight_names.append(weight_name)
                new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
    
    # Update model index
    new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    weight_map = dict(weight_map)
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


def _validate_staged_output(output_path):
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    shard_names = {
        os.path.basename(path)
        for path in glob(os.path.join(output_path, "*.safetensors"))
    }
    shard_tensors = {}
    for shard_name in shard_names:
        shard_path = os.path.join(output_path, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_tensors[shard_name] = set(shard.keys())

    for tensor_name, shard_name in weight_map.items():
        if shard_name not in shard_tensors or tensor_name not in shard_tensors[shard_name]:
            raise ValueError(
                f"Invalid staged index: {tensor_name} is missing from shard {shard_name}"
            )
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    
