import os
import json
from argparse import ArgumentParser
from glob import glob
from typing import Dict, Any
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant


class WeightConverter:
    def __init__(self, fp8_path: str, bf16_path: str):
        """
        Initialize the weight converter with input and output paths.

        Args:
            fp8_path (str): Path to the directory containing FP8 weights
            bf16_path (str): Path to save the converted BF16 weights
        """
        self.fp8_path = fp8_path
        self.bf16_path = bf16_path
        self.loaded_files: Dict[str, Dict[str, torch.Tensor]] = {}
        self.fp8_weight_names: list = []
        self.weight_map: Dict[str, str] = self._load_model_index()

    def _load_model_index(self) -> Dict[str, str]:
        """
        Load the model index file.

        Returns:
            Dict[str, str]: Weight mapping from the index file
        """
        model_index_file = os.path.join(self.fp8_path, "model.safetensors.index.json")
        with open(model_index_file, "r") as f:
            return json.load(f)["weight_map"]

    def _get_tensor(self, tensor_name: str) -> torch.Tensor:
        """
        Get a tensor from cache or load it from disk.

        Args:
            tensor_name (str): Name of the tensor to retrieve

        Returns:
            torch.Tensor: The requested tensor

        Raises:
            KeyError: If tensor doesn't exist in the safetensor file
        """
        file_name = self.weight_map[tensor_name]
        if file_name not in self.loaded_files:
            file_path = os.path.join(self.fp8_path, file_name)
            self.loaded_files[file_name] = load_file(file_path, device="cuda")
        return self.loaded_files[file_name][tensor_name]

    def _manage_memory(self):
        """
        Keep only the 2 most recently used files in memory.
        """
        if len(self.loaded_files) > 2:
            oldest_file = next(iter(self.loaded_files))
            del self.loaded_files[oldest_file]
            torch.cuda.empty_cache()

    def _process_weight(self, weight_name: str, weight: torch.Tensor) -> torch.Tensor:
        """
        Process a single weight tensor.

        Args:
            weight_name (str): Name of the weight tensor
            weight (torch.Tensor): The weight tensor to process

        Returns:
            torch.Tensor: Processed weight tensor
        """
        if weight_name.endswith("_scale_inv"):
            return None
        
        if weight.element_size() == 1:  # FP8 weight
            scale_inv_name = f"{weight_name}_scale_inv"
            try:
                scale_inv = self._get_tensor(scale_inv_name)
                self.fp8_weight_names.append(weight_name)
                return weight_dequant(weight, scale_inv)
            except KeyError:
                print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                return weight
        return weight

    def _save_model_index(self):
        """
        Save the updated model index file.
        """
        new_model_index_file = os.path.join(self.bf16_path, "model.safetensors.index.json")
        for weight_name in self.fp8_weight_names:
            scale_inv_name = f"{weight_name}_scale_inv"
            if scale_inv_name in self.weight_map:
                self.weight_map.pop(scale_inv_name)
                
        with open(new_model_index_file, "w") as f:
            json.dump({"metadata": {}, "weight_map": self.weight_map}, f, indent=2)

    def convert(self):
        """
        Convert FP8 weights to BF16 format.
        """
        torch.set_default_dtype(torch.bfloat16)
        os.makedirs(self.bf16_path, exist_ok=True)
        
        safetensor_files = sorted(glob(os.path.join(self.fp8_path, "*.safetensors")))
        
        for safetensor_file in tqdm(safetensor_files):
            file_name = os.path.basename(safetensor_file)
            current_state_dict = load_file(safetensor_file, device="cuda")
            self.loaded_files[file_name] = current_state_dict
            
            new_state_dict = {}
            for weight_name, weight in current_state_dict.items():
                processed_weight = self._process_weight(weight_name, weight)
                if processed_weight is not None:
                    new_state_dict[weight_name] = processed_weight
                    
            new_safetensor_file = os.path.join(self.bf16_path, file_name)
            save_file(new_state_dict, new_safetensor_file)
            
            self._manage_memory()
        
        self._save_model_index()


def main(fp8_path: str, bf16_path: str):
    """
    Main function to convert FP8 weights to BF16.

    Args:
        fp8_path (str): Input directory containing FP8 weights
        bf16_path (str): Output directory for BF16 weights
    """
    converter = WeightConverter(fp8_path, bf16_path)
    converter.convert()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)