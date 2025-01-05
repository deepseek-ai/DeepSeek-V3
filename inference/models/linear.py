import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..kernel import act_quant, weight_dequant, fp8_gemm

class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # ... (Linear implementation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (Linear forward implementation)

class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # ... (ColumnParallelLinear implementation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (ColumnParallelLinear forward implementation)

class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # ... (RowParallelLinear implementation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (RowParallelLinear forward implementation)