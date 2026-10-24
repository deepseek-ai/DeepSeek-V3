import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


def load_model():
    # These CPU tests exercise BF16 layers; fail if a CUDA-only kernel is used.
    kernel = types.ModuleType("kernel")

    def cuda_only(*args, **kwargs):
        raise AssertionError("Unexpected quantized CUDA kernel in BF16 MoE test")

    kernel.act_quant = cuda_only
    kernel.weight_dequant = cuda_only
    kernel.fp8_gemm = cuda_only
    path = Path(__file__).resolve().parents[1] / "inference" / "model.py"
    spec = importlib.util.spec_from_file_location("_moe_test_model", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"kernel": kernel, spec.name: module}):
        spec.loader.exec_module(module)
    return module


class MoELayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def setUp(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.addCleanup(torch.set_default_dtype, previous_dtype)
        self.model.world_size = 1
        self.model.rank = 0
        self.model.Linear.dtype = torch.bfloat16
        args = self.model.ModelArgs(
            dim=8, moe_inter_dim=4, n_routed_experts=4,
            n_shared_experts=1, n_activated_experts=2,
        )
        with torch.random.fork_rng():
            torch.manual_seed(42)
            self.moe = self.model.MoE(args)
            for parameter in self.moe.parameters():
                torch.nn.init.uniform_(parameter, -0.1, 0.1)

    def assert_layout_matches_contiguous(self, x):
        reference = x.detach().contiguous().requires_grad_(True)
        x = x.detach().requires_grad_(True)
        actual = self.moe(x)
        expected = self.moe(reference)
        self.assertEqual(actual.shape, x.shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        parameters = tuple(self.moe.parameters())
        actual_grads = torch.autograd.grad(actual.float().square().sum(), (x, *parameters), allow_unused=True)
        expected_grads = torch.autograd.grad(expected.float().square().sum(), (reference, *parameters), allow_unused=True)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            if actual_grad is None or expected_grad is None:
                self.assertIs(actual_grad, expected_grad)
            else:
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_transposed_batch_and_sequence(self):
        x = torch.arange(2 * 3 * 8).reshape(2, 3, 8).to(torch.bfloat16) / 100
        x = x.transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        self.assert_layout_matches_contiguous(x)

    def test_sliced_batch_and_sequence(self):
        x = torch.arange(4 * 6 * 8).reshape(4, 6, 8).to(torch.bfloat16) / 100
        x = x[::2, ::2, :]
        self.assertFalse(x.is_contiguous())
        self.assert_layout_matches_contiguous(x)

    def test_contiguous_batch(self):
        x = torch.arange(2 * 3 * 8).reshape(2, 3, 8).to(torch.bfloat16) / 100
        self.assert_layout_matches_contiguous(x)

    def test_flat_token_batch(self):
        x = torch.arange(6 * 8).reshape(6, 8).to(torch.bfloat16) / 100
        self.assert_layout_matches_contiguous(x)


if __name__ == "__main__":
    unittest.main()
