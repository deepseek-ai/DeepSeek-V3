import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path, stubs):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def torch_stubs():
    torch = types.ModuleType("torch")
    distributed = types.ModuleType("torch.distributed")
    torch.distributed = distributed
    torch.Tensor = object
    torch.bfloat16 = "bfloat16"
    torch.inference_mode = lambda: (lambda function: function)
    torch.set_default_dtype = lambda _dtype: None
    return torch, distributed


class GenerateEdgeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch, distributed = torch_stubs()
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = object
        safetensors = types.ModuleType("safetensors")
        safetensors_torch = types.ModuleType("safetensors.torch")
        safetensors_torch.load_model = lambda *_args, **_kwargs: None
        model = types.ModuleType("model")
        model.Transformer = object
        model.ModelArgs = object
        cls.module = load_module(
            "generate_under_test",
            ROOT / "inference" / "generate.py",
            {
                "torch": torch,
                "torch.distributed": distributed,
                "transformers": transformers,
                "safetensors": safetensors,
                "safetensors.torch": safetensors_torch,
                "model": model,
            },
        )

    def test_generate_rejects_empty_batch(self):
        model = types.SimpleNamespace(max_seq_len=16)
        with self.assertRaisesRegex(ValueError, "At least one prompt"):
            self.module.generate(model, [], 1, 0)

    def test_generate_rejects_empty_token_sequence(self):
        model = types.SimpleNamespace(max_seq_len=16)
        with self.assertRaisesRegex(ValueError, "at least one token"):
            self.module.generate(model, [[]], 1, 0)

    def test_generate_rejects_negative_token_limit(self):
        model = types.SimpleNamespace(max_seq_len=16)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.module.generate(model, [[1]], -1, 0)

    def test_generate_rejects_prompt_over_model_limit(self):
        model = types.SimpleNamespace(max_seq_len=2)
        with self.assertRaisesRegex(ValueError, "maximum sequence length"):
            self.module.generate(model, [[1, 2, 3]], 1, 0)

    def test_batch_mode_rejects_file_without_prompts(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as input_file:
            input_file.write("  \n\n")
            input_file.flush()
            with self.assertRaisesRegex(ValueError, "no non-empty prompts"):
                self.module.main("unused", "unused", input_file.name, interactive=False)


class Fp8ConversionEdgeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch, _ = torch_stubs()
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
        safetensors = types.ModuleType("safetensors")
        safetensors_torch = types.ModuleType("safetensors.torch")
        safetensors_torch.load_file = lambda *_args, **_kwargs: {}
        safetensors_torch.save_file = lambda *_args, **_kwargs: None
        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda values: values
        kernel = types.ModuleType("kernel")
        kernel.weight_dequant = lambda weight, _scale: weight
        cls.module = load_module(
            "fp8_cast_bf16_under_test",
            ROOT / "inference" / "fp8_cast_bf16.py",
            {
                "torch": torch,
                "safetensors": safetensors,
                "safetensors.torch": safetensors_torch,
                "tqdm": tqdm,
                "kernel": kernel,
            },
        )

    def test_rejects_in_place_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be different"):
                self.module.main(directory, directory)

    def test_preserves_model_index_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "output"
            source.mkdir()
            index = {
                "metadata": {"total_size": 123},
                "weight_map": {"model.weight": "model-00001.safetensors"},
            }
            (source / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            self.module.main(str(source), str(output))

            converted_index = json.loads(
                (output / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(converted_index, index)


if __name__ == "__main__":
    unittest.main()
