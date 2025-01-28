import os
import json
import logging
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import OrderedDict

from tqdm import tqdm
import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_SIZE = 2  # Number of safetensors files to keep in memory
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_WEIGHT_TYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

def validate_paths(fp8_path: Path, bf16_path: Path) -> None:
    """Validate input and output paths."""
    if not fp8_path.is_dir():
        raise ValueError(f"Input directory {fp8_path} does not exist")
    if not (fp8_path / "model.safetensors.index.json").exists():
        raise FileNotFoundError("Missing model index file in input directory")
    
    bf16_path.mkdir(parents=True, exist_ok=True)
    if not os.access(bf16_path, os.W_OK):
        raise PermissionError(f"No write permission for output directory {bf16_path}")

def load_model_index(fp8_path: Path) -> Tuple[Dict, Dict]:
    """Load and validate model index file."""
    index_path = fp8_path / "model.safetensors.index.json"
    try:
        with open(index_path, "r") as f:
            model_index = json.load(f)
        return model_index, model_index["weight_map"].copy()
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Invalid model index file: {str(e)}")
        raise

def process_weight(
    weight_name: str,
    weight: torch.Tensor,
    weight_map: Dict[str, str],
    file_cache: OrderedDict,
    fp8_path: Path,
    fp8_weight_names: list
) -> Optional[torch.Tensor]:
    """Process a single weight tensor."""
    if weight_name.endswith("_scale_inv"):
        return None

    if weight.dtype in VALID_WEIGHT_TYPES and weight.element_size() == 1:
        return handle_fp8_weight(weight_name, weight, weight_map, file_cache, fp8_path, fp8_weight_names)
    
    return weight.clone()

def handle_fp8_weight(
    weight_name: str,
    weight: torch.Tensor,
    weight_map: Dict[str, str],
    file_cache: OrderedDict,
    fp8_path: Path,
    fp8_weight_names: list
) -> torch.Tensor:
    """Handle FP8 weight conversion to BF16."""
    scale_inv_name = f"{weight_name}_scale_inv"
    try:
        scale_inv = load_tensor_from_cache(scale_inv_name, weight_map, file_cache, fp8_path)
        fp8_weight_names.append(weight_name)
        return weight_dequant(weight, scale_inv)
    except KeyError:
        logger.warning(f"Missing scale_inv tensor for {weight_name}, using original weight")
        return weight.clone()
    except Exception as e:
        logger.error(f"Error processing {weight_name}: {str(e)}")
        raise

def load_tensor_from_cache(
    tensor_name: str,
    weight_map: Dict[str, str],
    file_cache: OrderedDict,
    fp8_path: Path
) -> torch.Tensor:
    """Load tensor from cached files or disk."""
    if tensor_name not in weight_map:
        raise KeyError(f"Tensor {tensor_name} not found in weight map")
    
    file_name = weight_map[tensor_name]
    if file_name not in file_cache:
        load_file_to_cache(file_name, file_cache, fp8_path)
    
    return file_cache[file_name][tensor_name]

def load_file_to_cache(file_name: str, file_cache: OrderedDict, fp8_path: Path) -> None:
    """Load safetensors file into cache with LRU eviction."""
    file_path = fp8_path / file_name
    try:
        file_cache[file_name] = load_file(str(file_path), device=TORCH_DEVICE)
        file_cache.move_to_end(file_name)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {str(e)}")
        raise
    
    while len(file_cache) > CACHE_SIZE:
        oldest = next(iter(file_cache))
        del file_cache[oldest]
        torch.cuda.empty_cache()

def process_safetensor_file(
    file_path: Path,
    bf16_path: Path,
    weight_map: Dict[str, str],
    file_cache: OrderedDict,
    fp8_path: Path,
    fp8_weight_names: list
) -> None:
    """Process a single safetensors file."""
    try:
        current_state_dict = load_file(str(file_path), device=TORCH_DEVICE)
        file_cache[file_path.name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in tqdm(current_state_dict.items(), 
                                      desc=f"Processing {file_path.name}", 
                                      leave=False):
            processed_weight = process_weight(
                weight_name, weight, weight_map, 
                file_cache, fp8_path, fp8_weight_names
            )
            if processed_weight is not None:
                new_state_dict[weight_name] = processed_weight
        
        save_converted_file(new_state_dict, file_path.name, bf16_path)
    except Exception as e:
        logger.error(f"Failed to process {file_path.name}: {str(e)}")
        raise

def save_converted_file(state_dict: Dict[str, torch.Tensor], filename: str, bf16_path: Path) -> None:
    """Save converted state dict to file."""
    output_path = bf16_path / filename
    try:
        save_file(state_dict, str(output_path), metadata={"converted": "fp8_to_bf16"})
        logger.debug(f"Saved converted file: {filename}")
    except Exception as e:
        logger.error(f"Failed to save {filename}: {str(e)}")
        raise

def update_model_index(weight_map: Dict[str, str], fp8_weight_names: list, bf16_path: Path) -> None:
    """Update model index file with converted weights."""
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            del weight_map[scale_inv_name]
    
    index_path = bf16_path / "model.safetensors.index.json"
    try:
        with open(index_path, "w") as f:
            json.dump({
                "metadata": {"conversion": "fp8_to_bf16"},
                "weight_map": weight_map
            }, f, indent=2)
        logger.info(f"Updated model index saved to {index_path}")
    except Exception as e:
        logger.error(f"Failed to save model index: {str(e)}")
        raise

def main(fp8_path: Path, bf16_path: Path) -> None:
    """Main conversion function."""
    torch.set_default_dtype(torch.bfloat16)
    validate_paths(fp8_path, bf16_path)
    
    try:
        model_index, weight_map = load_model_index(fp8_path)
        file_cache = OrderedDict()
        fp8_weight_names = []

        safetensor_files = sorted(fp8_path.glob("*.safetensors"))
        for safetensor_file in tqdm(safetensor_files, desc="Processing files"):
            process_safetensor_file(
                safetensor_file, bf16_path, 
                weight_map, file_cache, fp8_path, 
                fp8_weight_names
            )

        update_model_index(weight_map, fp8_weight_names, bf16_path)
        logger.info(f"Successfully converted {len(fp8_weight_names)} weights to BF16")

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert FP8 model weights to BF16 format")
    parser.add_argument(
        "--input-fp8-hf-path",
        type=Path,
        required=True,
        help="Path to input directory with FP8 weights"
    )
    parser.add_argument(
        "--output-bf16-hf-path",
        type=Path,
        required=True,
        help="Output directory for converted BF16 weights"
    )
    
    args = parser.parse_args()
    
    try:
        main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    except Exception as e:
        logger.critical(f"Fatal error during conversion: {str(e)}")
        exit(1)
