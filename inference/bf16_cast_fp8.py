import json
import os
import re
from argparse import ArgumentParser
from glob import glob

import torch
from auto_fp8 import BaseQuantizeConfig
from kernel import fp8_weight_block_wise_quant
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig


# Helper function to get tensor from the correct file
def has_tensor(weight_map, loaded_files, fp8_path, tensor_name):
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


def find_ignored(regex_pat, weight_name):
    searched = regex_pat.search(weight_name)
    if searched is not None:
        print(f"find : {searched.string}")
        return searched.string
    return None


def find_one_ignored(regex_pat_list, weight_name):
    for regex_pat in regex_pat_list:
        searched = find_ignored(regex_pat, weight_name)
        if searched is not None:
            return searched
    return None


quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="dynamic",
    ignore_patterns=[".*lm_head", ".*gate"],
)


def main(bf16_path, fp8_path, ref_weights_scale_inv_map=None):
    """
    Quantize BF16 to FP8 (OCP E4M3) and saves the converted weights.

    This function reads BF16 weights from the specified directory, converts them to FP8 (OCP E4M3),
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    bf16_path (str): The path to the directory containing the BF16 weights and model index file.
    fp8_path (str): The path to the directory where the converted FP8 (OCP E4M3) weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the BF16 weights are stored in safetensor files.
    - The function update the model index file to add references to scale_inv tensors.
    """
    # torch.set_default_dtype(torch.bfloat16)
    os.makedirs(fp8_path, exist_ok=True)

    model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    bf16_weight_names = []

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if (
                find_one_ignored(quantize_config.ignore_patterns, weight_name)
                is not None
            ):
                continue
            elif weight.element_size() == 2:  # BF16 weight

                if (
                    ref_weights_scale_inv_map is not None
                    and ref_weights_scale_inv_map.get(weight_name, None) is None
                ):
                    print(f"skipping {weight_name} ...")
                    continue
                    pass

                scale_inv_name = f"{weight_name}_scale_inv"
                bf16_weight_names.append(weight_name)
                fp8_weight, scale_inv = fp8_weight_block_wise_quant(weight)
                new_state_dict[weight_name] = fp8_weight
                new_state_dict[scale_inv_name] = scale_inv
            else:
                new_state_dict[weight_name] = weight
            pass

        new_safetensor_file = os.path.join(fp8_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index
    new_model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    for weight_name in bf16_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.insert(scale_inv_name)
        pass

    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    pass


def update_quant_model_config(bf16_cast_fp8_path):
    cfg = AutoConfig.from_pretrained(bf16_cast_fp8_path)

    static_q_dict = {
        "quantization_config": {
            "activation_scheme": quantize_config.activation_scheme,
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "ignored_layers": quantize_config.re_ignore_patterns,
        }
    }

    cfg.update(static_q_dict)
    cfg.to_json_file(os.path.join(bf16_cast_fp8_path, "config.json.bak"))
    pass


def read_weight_inv_list(fp8_path):
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    weights_with_scale_inv_map = {}

    loaded_files = {}
    fp8_weight_names = []

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
            elif weight.element_size() == 1:  # FP8 weight:
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = has_tensor(
                        weight_map, loaded_files, fp8_path, scale_inv_name
                    )
                    fp8_weight_names.append(weight_name)
                    weights_with_scale_inv_map[weight_name] = weight_map[scale_inv_name]
                except KeyError:
                    print(
                        f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion"
                    )
                    new_state_dict[weight_name] = weight
                pass
            pass

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
        pass

    weights_with_scale_inv = os.path.join(
        fp8_path, "weight_with_scale_inv_map.index.json"
    )
    with open(weights_with_scale_inv, "w") as f:
        json.dump(
            {"metadata": {}, "weight_with_scale_inv_map": weights_with_scale_inv_map},
            f,
            indent=2,
        )
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=False)
    parser.add_argument("--output-fp8-hf-path", type=str, required=False)
    parser.add_argument("--input-fp8-hf-path", type=str, required=False)
    parser.add_argument("--input-new-fp8-hf-path", type=str, required=False)
    args = parser.parse_args()

    if (
        args.input_fp8_hf_path is not None
        and args.input_bf16_hf_path is None
        and args.output_fp8_hf_path is None
    ):
        read_weight_inv_list(args.input_fp8_hf_path)
    elif args.input_new_fp8_hf_path is not None:
        update_quant_model_config(args.input_new_fp8_hf_path)
        pass
    else:
        assert (
            args.input_bf16_hf_path is not None and args.output_fp8_hf_path is not None
        )
        if args.input_fp8_hf_path is not None:
            weights_with_scale_inv = os.path.join(
                args.input_fp8_hf_path, "weight_with_scale_inv_map.index.json"
            )
            with open(weights_with_scale_inv, "r") as f:
                model_index = json.load(f)
                pass
            weight_with_scale_inv_map = model_index["weight_with_scale_inv_map"]
            pass
        main(
            args.input_bf16_hf_path,
            args.output_fp8_hf_path,
            ref_weights_scale_inv_map=weight_with_scale_inv_map,
        )
