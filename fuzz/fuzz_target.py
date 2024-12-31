import atheris
import sys
import os
import json
import torch
from safetensors.torch import load_file, save_file
from kernel import weight_dequant

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    try:
        fp8_path = fdp.ConsumeUnicodeNoSurrogates(50)
        bf16_path = fdp.ConsumeUnicodeNoSurrogates(50)
        torch.set_default_dtype(torch.bfloat16)
        os.makedirs(bf16_path, exist_ok=True)
        model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]
        for tensor_name in weight_map:
            file_name = weight_map[tensor_name]
            file_path = os.path.join(fp8_path, file_name)
            load_file(file_path, device="cuda")
    except Exception as e:
        pass

atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
