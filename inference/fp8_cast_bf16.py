import json
import math
import os
import shutil
import tempfile
from argparse import ArgumentParser

import torch
from glob import glob
from safetensors.torch import load_file, safe_open, save_file
from tqdm import tqdm

from kernel import weight_dequant


def _remove_path(path):
    """Remove a file, symlink, or directory if it exists."""
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.unlink(path)


def _paths_overlap(source, destination):
    try:
        common = os.path.commonpath((source, destination))
    except ValueError:
        return False
    return common == source or common == destination


def _validate_source_index(source, weight_map):
    """Reject malformed shard paths before opening files referenced by the index."""
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index must contain a non-empty weight_map")

    source_root = os.path.realpath(source)
    shard_names = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("model index contains an invalid tensor name")
        if (
            not isinstance(shard_name, str)
            or not shard_name
            or os.path.basename(shard_name) != shard_name
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError(
                f"Invalid safetensors shard path in model index for {tensor_name}"
            )

        shard_path = os.path.realpath(os.path.join(source_root, shard_name))
        try:
            contained = os.path.commonpath((source_root, shard_path)) == source_root
        except ValueError:
            contained = False
        if not contained or not os.path.isfile(shard_path):
            raise ValueError(
                f"Indexed safetensors shard is missing or outside the input directory: {shard_name}"
            )
        shard_names.add(shard_name)

    return sorted(shard_names)


def _publish_staged_directory(staging_path, destination):
    """Publish a completed sibling directory to a destination that must not exist."""
    # Recheck after conversion so this operation never intentionally replaces a
    # pre-existing output. Callers must use a fresh output path; concurrent
    # writers targeting the same name are not supported.
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Output already exists: {destination}. Choose a new output directory; "
            "existing outputs are never overwritten."
        )
    # Staging is created under the destination's parent, keeping the rename on
    # the same filesystem. The destination becomes visible only after the full
    # conversion and index validation have completed.
    os.replace(staging_path, destination)


def main(fp8_path, bf16_path):
    """
    Convert FP8 safetensors weights to BF16 and publish a validated output directory.

    The output path must not already exist. This avoids a non-atomic directory
    replacement window and ensures that a failed conversion cannot damage an
    existing output directory.

    Raises:
        KeyError: If a required scale_inv tensor is missing.
        ValueError: If the input index or staged output is invalid, or the input
            and output directory paths overlap.
        FileExistsError: If the output directory already exists.
    """
    source = os.path.realpath(os.path.abspath(fp8_path))
    requested_destination = os.path.abspath(bf16_path)
    destination = os.path.realpath(requested_destination)

    if not os.path.isdir(source):
        raise ValueError(f"Input directory does not exist: {fp8_path}")
    if os.path.lexists(requested_destination) or os.path.lexists(destination):
        raise FileExistsError(
            f"Output already exists: {bf16_path}. Choose a new output directory; "
            "existing outputs are never overwritten."
        )
    if _paths_overlap(source, destination):
        raise ValueError("Input and output directories must not contain or overlap each other")

    index_path = os.path.join(source, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as index_file:
        model_index = json.load(index_file)

    weight_map = model_index.get("weight_map")
    shard_names = _validate_source_index(source, weight_map)
    metadata = model_index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("model index metadata must be a JSON object")

    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    staging_path = None
    old_dtype = torch.get_default_dtype()

    try:
        staging_path = tempfile.mkdtemp(
            prefix=f".{os.path.basename(destination)}.staging-",
            dir=parent,
        )
        torch.set_default_dtype(torch.bfloat16)
        _convert_to_directory(source, weight_map, shard_names, staging_path, metadata)
        _validate_staged_output(staging_path)
        _publish_staged_directory(staging_path, destination)
        staging_path = None
    except BaseException:
        if staging_path is not None and os.path.lexists(staging_path):
            try:
                _remove_path(staging_path)
            except OSError:
                # Preserve the original conversion/publication error. A leftover
                # staging path, if cleanup is denied by the filesystem, is hidden
                # and can be diagnosed separately.
                pass
        raise
    finally:
        torch.set_default_dtype(old_dtype)


def _is_fp8_tensor(tensor):
    # Storage width alone is not a valid dtype test: INT8/UINT8 tensors must not
    # be sent to the FP8 dequantization kernel.
    return str(tensor.dtype).startswith("torch.float8_")


def _convert_to_directory(source, weight_map, shard_names, output_path, source_metadata):
    loaded_files = {}
    fp8_weight_names = []
    total_size_bytes = 0

    def get_tensor(tensor_name):
        """Retrieve a tensor from the indexed source shards."""
        if tensor_name not in weight_map:
            raise KeyError(f"Missing required tensor in model index: {tensor_name}")
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            loaded_files[file_name] = load_file(
                os.path.join(source, file_name), device="cuda"
            )
        if tensor_name not in loaded_files[file_name]:
            raise KeyError(f"Tensor {tensor_name} is missing from shard {file_name}")
        return loaded_files[file_name][tensor_name]

    for file_name in tqdm(shard_names):
        current_state_dict = load_file(os.path.join(source, file_name), device="cuda")
        loaded_files[file_name] = current_state_dict
        new_state_dict = {}

        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            if _is_fp8_tensor(weight):
                scale_inv_name = f"{weight_name}_scale_inv"
                scale_inv = get_tensor(scale_inv_name)
                fp8_weight_names.append(weight_name)
                new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
            else:
                new_state_dict[weight_name] = weight

        output_shard = os.path.join(output_path, file_name)
        save_file(new_state_dict, output_shard)
        total_size_bytes += sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in new_state_dict.values()
        )

        # Retain a small cache for cross-shard scale lookups.
        while len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_weight_map = dict(weight_map)
    for weight_name in fp8_weight_names:
        output_weight_map.pop(f"{weight_name}_scale_inv", None)

    output_metadata = dict(source_metadata)
    # FP8 weights become BF16 and scale tensors are removed, so the old byte
    # count is stale. Preserve the field only when the source provided it.
    if "total_size" in output_metadata:
        output_metadata["total_size"] = total_size_bytes

    output_index = os.path.join(output_path, "model.safetensors.index.json")
    with open(output_index, "w", encoding="utf-8") as index_file:
        json.dump(
            {"metadata": output_metadata, "weight_map": output_weight_map},
            index_file,
            indent=2,
        )


def _validate_staged_output(output_path):
    """Check every indexed tensor exists in a staged safetensors shard."""
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as index_file:
        staged_index = json.load(index_file)

    weight_map = staged_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid staged index: weight_map is missing or not an object")

    shard_tensors = {}
    for shard_path in glob(os.path.join(output_path, "*.safetensors")):
        shard_name = os.path.basename(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_tensors[shard_name] = set(shard.keys())

    for tensor_name, shard_name in weight_map.items():
        if (
            not isinstance(shard_name, str)
            or os.path.basename(shard_name) != shard_name
            or shard_name not in shard_tensors
            or tensor_name not in shard_tensors[shard_name]
        ):
            raise ValueError(
                f"Invalid staged index: {tensor_name} is missing from shard {shard_name}"
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
