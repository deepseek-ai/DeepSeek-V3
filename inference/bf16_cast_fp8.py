import os
import json
import re
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_quant

# Layers that should not be quantized (remain in BF16)
SKIP_QUANT_PATTERNS = [
    r".*\.layernorm\.weight$",
    r".*\.norm\.weight$",
    r".*input_layernorm\.weight$",
    r".*post_attention_layernorm\.weight$",
    r".*\.kv_a_layernorm\.weight$",
    r".*\.q_a_layernorm\.weight$",
    r".*\.embed_tokens\.weight$",
    r".*\.head\.weight$",
    r".*lm_head\.weight$",
    r".*\.eh_proj\.weight$",
    r".*\.gate\.e_score_correction_bias$",
    r".*\.gate\.weight$"
]

def should_skip_quantization(weight_name):
    """Check if weight name matches any pattern in the skip list"""
    return any(re.match(pattern, weight_name) for pattern in SKIP_QUANT_PATTERNS)

def main(bf16_path, fp8_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(fp8_path, exist_ok=True)

    # Get list of safetensor files
    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()

    # Load model index if it exists
    model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    if os.path.exists(model_index_file):
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]
    else:
        # Create a new weight map if there's no index file
        weight_map = {}

    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            # Skip weights that should not be quantized
            if should_skip_quantization(weight_name) or weight.dim() != 2:
                new_state_dict[weight_name] = weight
            else:
                # Quantize weights to FP8
                fp8_weight, scale_inv = weight_quant(weight)
                new_state_dict[weight_name] = fp8_weight
                scale_inv_name = f"{weight_name}_scale_inv"
                new_state_dict[scale_inv_name] = scale_inv
                fp8_weight_names.append(weight_name)

                # Update weight map
                if weight_name in weight_map:
                    weight_map[scale_inv_name] = file_name

        new_safetensor_file = os.path.join(fp8_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index
    new_model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True)
    parser.add_argument("--output-fp8-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_bf16_hf_path, args.output_fp8_hf_path)