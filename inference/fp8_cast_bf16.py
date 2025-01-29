import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant


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

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)

    try:
        os.makedirs(bf16_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {bf16_path}: {e}")
        return

    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")

    if not os.path.isfile(model_index_file):
        print(f"Error: Model index file '{model_index_file}' does not exist.")
        return

    try:
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading model index file '{model_index_file}': {e}")
        return

    weight_map = model_index.get("weight_map", {})

    loaded_files = {}
    fp8_weight_names = []

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
        try:
            file_name = weight_map[tensor_name]
        except KeyError:
            raise KeyError(f"Tensor '{tensor_name}' not found in weight map.")

        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            try:
                loaded_files[file_name] = load_file(file_path, device="cuda")
            except (FileNotFoundError, OSError) as e:
                raise FileNotFoundError(f"Error loading file '{file_path}': {e}")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)

        try:
            current_state_dict = load_file(safetensor_file, device="cuda")
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading safetensor file '{safetensor_file}': {e}")
            continue

        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:
                scale_inv_name = f"{weight_name}_scale_inv"
                try:

                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(
                        f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion"
                    )
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(bf16_path, file_name)

        try:
            save_file(new_state_dict, new_safetensor_file)
        except (OSError, RuntimeError) as e:
            print(f"Error saving safetensor file '{new_safetensor_file}': {e}")
            continue

        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")

    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)

    try:
        with open(new_model_index_file, "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error writing new model index file '{new_model_index_file}': {e}")
        return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.input_fp8_hf_path):
        print(
            f"Error: Input FP8 path '{args.input_fp8_hf_path}' is not a valid directory."
        )
    elif not os.path.isdir(args.output_bf16_hf_path):
        print(
            f"Error: Output BF16 path '{args.output_bf16_hf_path}' is not a valid directory."
        )
    else:
        main(args.input_fp8_hf_path, args.output_bf16_hf_path)
