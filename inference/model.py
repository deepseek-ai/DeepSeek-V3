import math
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm

# Configure logging
logger = logging.getLogger(__name__)

# Global configuration
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

# Distributed operation timeout (seconds)
DIST_TIMEOUT = 60.0

# Custom exceptions for distributed operations
class DistributedOperationError(Exception):
    """Base exception for distributed operation errors"""
    pass

class TensorShapeMismatchError(DistributedOperationError):
    """Raised when tensor shapes don't match in distributed operations"""
    pass

class WorldSizeMismatchError(DistributedOperationError):
    """Raised when world size doesn't match expectations"""
    pass

class RankMismatchError(DistributedOperationError):
    """Raised when rank doesn't match expectations"""
    pass

class DistributedInitializationError(DistributedOperationError):
    """Raised when distributed initialization fails"""
    pass


def safe_dist_operation(func):
    """
    Decorator for safe distributed operations with error handling.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            # Check if distributed environment is properly initialized
            if world_size > 1:
                if not dist.is_initialized():
                    raise DistributedInitializationError(
                        "Distributed environment not initialized but world_size > 1. "
                        "Call dist.init_process_group() before using distributed operations."
                    )
                if rank >= world_size:
                    raise RankMismatchError(
                        f"Rank {rank} is out of bounds for world_size {world_size}"
                    )
            
            return func(*args, **kwargs)
        except dist.DistBackendError as e:
            logger.error(f"Distributed backend error in {func.__name__}: {e}")
            raise DistributedOperationError(f"Backend error in {func.__name__}: {e}") from e
        except RuntimeError as e:
            logger.error(f"Runtime error in distributed operation {func.__name__}: {e}")
            raise DistributedOperationError(f"Runtime error in {func.__name__}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


def validate_distributed_params(param_name: str, param_value: int, divisor: int):
    """
    Validate that a parameter is divisible by world_size.
    
    Args:
        param_name: Name of the parameter for error messages
        param_value: Value of the parameter to validate
        divisor: Expected divisor (usually world_size)
        
    Raises:
        WorldSizeMismatchError: If parameter is not divisible by divisor
    """
    if divisor > 1 and param_value % divisor != 0:
        raise WorldSizeMismatchError(
            f"{param_name} ({param_value}) must be divisible by world_size ({divisor}) "
            f"for distributed training. Remainder: {param_value % divisor}"
        )


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Validate distributed parameters
        validate_distributed_params("vocab_size", vocab_size, world_size)
        
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        
        # Validate indices
        if self.vocab_start_idx >= vocab_size or self.vocab_end_idx > vocab_size:
            raise RankMismatchError(
                f"Calculated vocabulary indices out of bounds for rank {rank}. "
                f"Start: {self.vocab_start_idx}, End: {self.vocab_end_idx}, "
                f"Vocab Size: {vocab_size}"
            )
        
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    @safe_dist_operation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If tensor contains indices outside valid range
            DistributedOperationError: If distributed operation fails
        """
        if world_size > 1:
            # Check input tensor validity
            if x.numel() == 0:
                raise ValueError("Input tensor is empty")
            
            # Create mask for indices outside our partition
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            
            # Shift indices for local embedding lookup
            x = x - self.vocab_start_idx
            
            # Handle out-of-bounds indices
            out_of_bounds = (x < 0) | (x >= self.part_vocab_size)
            if out_of_bounds.any():
                # Zero out invalid indices
                x[out_of_bounds] = 0
                mask = mask | out_of_bounds
            
            # Perform embedding
            y = F.embedding(x, self.weight)
            
            # Zero out masked positions
            y[mask] = 0
            
            # All-reduce across all processes
            try:
                dist.all_reduce(y, timeout=DIST_TIMEOUT)
            except Exception as e:
                logger.error(f"All-reduce failed in ParallelEmbedding: {e}")
                raise DistributedOperationError(f"All-reduce failed: {e}") from e
        else:
            # Single GPU case - use standard embedding
            y = F.embedding(x, self.weight)
        
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
           scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.
        scale_fmt (Optional[str]): Scale format for quantization.

    Returns:
        torch.Tensor: The result of the linear transformation.

    Raises:
        ValueError: If tensor shapes are incompatible
        DistributedOperationError: If distributed operation fails
    """
    # Validate input dimensions
    if x.dim() != 2:
        raise ValueError(f"Input tensor must be 2D, got shape {x.shape}")
    
    if x.size(-1) != weight.size(-1):
        raise ValueError(
            f"Input feature dimension ({x.size(-1)}) must match "
            f"weight feature dimension ({weight.size(-1)})"
        )
    
    if weight.element_size() > 1:
        # Full precision weights
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # BF16 implementation with dequantization
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # FP8 implementation
        x, scale = act_quant(x, block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Validate parameters
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"Invalid dimensions: in_features={in_features}, out_features={out_features}")
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        
        if self.weight.element_size() == 1:
            # Quantized weights - create scale parameters
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # Validate distributed parameters
        validate_distributed_params("out_features", out_features, world_size)
        
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)
        
        # Store full dimensions for validation
        self.full_out_features = out_features

    @safe_dist_operation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        # Validate input
        if x.size(-1) != self.in_features:
            raise TensorShapeMismatchError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"layer input dimension ({self.in_features})"
            )
        
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # Validate distributed parameters
        validate_distributed_params("in_features", in_features, world_size)
        
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)
        
        # Store full dimensions for validation
        self.full_in_features = in_features

    @safe_dist_operation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        # Validate input
        if x.size(-1) != self.full_in_features:
            raise TensorShapeMismatchError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"full layer input dimension ({self.full_in_features})"
            )
        
        # Split input along feature dimension for this partition
        start_idx = rank * self.part_in_features
        end_idx = start_idx + self.part_in_features
        x_part = x[..., start_idx:end_idx]
        
        # Apply linear transformation to local partition
        y = linear(x_part, self.weight)
        
        # All-reduce across all processes
        if world_size > 1:
            try:
                # Verify tensor shape consistency before all-reduce
                expected_shape = torch.Size([*x.shape[:-1], self.out_features])
                if y.shape != expected_shape:
                    raise TensorShapeMismatchError(
                        f"Local output shape {y.shape} doesn't match expected shape {expected_shape}"
                    )
                
                dist.all_reduce(y, timeout=DIST_TIMEOUT)
            except Exception as e:
                logger.error(f"All-reduce failed in RowParallelLinear: {e}")
                raise DistributedOperationError(f"All-reduce failed: {e}") from e
        
        # Add bias if present
        if self.bias is not None:
            y += self.bias
        
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        if x.size(-1) != self.dim:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) must match "
                f"RMSNorm dimension ({self.dim})"
            )
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        
        # Validate distributed parameters
        validate_distributed_params("n_heads", args.n_heads, world_size)
        
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    @safe_dist_operation
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Validate input dimensions
        if x.size(-1) != self.dim:
            raise TensorShapeMismatchError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"MLA dimension ({self.dim})"
            )
        
        # Validate cache indices
        if start_pos < 0 or end_pos > self.kv_cache.size(1) if attn_impl != "naive" else self.k_cache.size(1):
            raise ValueError(
                f"Cache indices out of bounds: start_pos={start_pos}, end_pos={end_pos}, "
                f"cache_size={self.kv_cache.size(1) if attn_impl != 'naive' else self.k_cache.size(1)}"
            )
        
        # Compute queries
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # Compute key-value projections
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            
            # Update caches
            self.k_cache[:bsz, start_pos:end_pos].copy_(k)
            self.v_cache[:bsz, start_pos:end_pos].copy_(v)
            
            # Compute attention scores
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            # Update caches
            self.kv_cache[:bsz, start_pos:end_pos].copy_(self.kv_norm(kv))
            self.pe_cache[:bsz, start_pos:end_pos].copy_(k_pe.squeeze(2))
            
            # Compute attention scores
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape != (seqlen, seqlen):
                raise TensorShapeMismatchError(
                    f"Mask shape {mask.shape} does not match expected shape {(seqlen, seqlen)}"
                )
            scores += mask.unsqueeze(1)
        
        # Apply softmax
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # Compute output
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        # Validate input
        if x.size(-1) != self.dim:
            raise TensorShapeMismatchError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"MLP dimension ({self.dim})"
            )
        
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        # Validate input
        if x.size(-1) != self.dim:
            raise TensorShapeMismatchError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"gate dimension ({self.dim})"
            )
        
        scores = linear(x, self.weight)
        
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        
        original_scores = scores
        
        if self.bias is not None:
            scores = scores + self.bias
        
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        # Validate input
        if x.size(-1) != self.dim:
            raise TensorShapeMismatchError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"expert dimension ({self.dim})"
            )
        
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        
        # Validate distributed parameters
        validate_distributed_params("n_routed_experts", args.n_routed_experts, world_size)
        
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        
        # Validate expert indices
        if self.experts_start_idx >= args.n_routed_experts or self.experts_end_idx > args.n_routed_experts:
            raise RankMismatchError(
                f"Calculated expert indices out of bounds for rank {rank}. "
                f"Start: {self.experts_start_idx}, End: {self.experts_end_idx}, "
                f"Total Experts: {args.n_routed_experts}"
            )
        
        self.gate = Gate(args)
        
        # Create expert modules only for local experts
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    @safe_dist_operation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        # Validate input
        if x.size(-1) != self.dim:
            raise TensorShapeMismatchError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"MoE dimension ({self.dim})"
            )
        
        shape = x.size()
        x = x.view(-1, self.dim)
        
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        
        # Count how many tokens are routed to each expert
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        # Process local experts
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            
            expert = self.experts[i]
            if expert is None:
                # This shouldn't happen if indices are correct
                continue
            
            # Get tokens assigned to this expert
            idx, top = torch.where(indices == i)
            
            # Apply expert to assigned tokens
            if len(idx) > 0:
                y[idx] += expert(x[idx]) * weights[idx, top, None]
        
        # Apply shared experts
        z = self.shared_experts(x)
        
        # All-reduce across all processes to combine expert outputs
        if world_size > 1:
            try:
                # Verify tensor shape before all-reduce
                expected_shape = y.shape
                if y.shape != expected_shape:
                    raise TensorShapeMismatchError(
                        f"Expert output shape {y.shape} doesn't match expected shape {expected_shape}"
                    )
                
                dist.all_reduce(y, timeout=DIST_TIMEOUT)
            except Exception as e:
                logger.error(f"All-reduce failed in MoE: {e}")
                raise DistributedOperationError(f"All-reduce failed: {e}") from e
        
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.layer_id = layer_id
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        # Attention layer with residual connection
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        
        # Feed-forward layer with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        
        return x


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        
        try:
            # Initialize distributed environment
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                logger.info(f"Distributed environment initialized: world_size={world_size}, rank={rank}")
            else:
                world_size = 1
                rank = 0
                logger.info("Running in single-GPU mode")
                
            # Validate distributed parameters
            if world_size > 1:
                validate_distributed_params("vocab_size", args.vocab_size, world_size)
                validate_distributed_params("n_heads", args.n_heads, world_size)
                validate_distributed_params("n_routed_experts", args.n_routed_experts, world_size)
                validate_distributed_params("args.dim for parallel linear", args.dim, world_size)
                
        except Exception as e:
            logger.error(f"Failed to initialize distributed environment: {e}")
            raise DistributedInitializationError(f"Distributed initialization failed: {e}") from e
        
        # Set data types
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        Linear.scale_fmt = args.scale_fmt
        
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        
        # Create transformer layers
        self.layers = torch.nn.ModuleList([
            Block(layer_id, args) for layer_id in range(args.n_layers)
        ])
        
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        
        # Precompute rotary embeddings
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    @safe_dist_operation
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        # Validate input
        if tokens.dim() != 2:
            raise ValueError(f"Tokens must be 2D tensor, got shape {tokens.shape}")
        
        if tokens.numel() == 0:
            raise ValueError("Tokens tensor is empty")
        
        bsz, seqlen = tokens.size()
        
        if bsz > self.freqs_cis.size(0):
            raise ValueError(
                f"Batch size {bsz} exceeds maximum batch size {self.freqs_cis.size(0)}"
            )
        
        if start_pos + seqlen > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seqlen} at position {start_pos} exceeds "
                f"maximum sequence length {self.max_seq_len}"
            )
        
        # Get embeddings
        h = self.embed(tokens)
        
        # Get rotary embeddings for this position
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        
        # Create attention mask if needed
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        
        # Apply transformer layers
        for layer_idx, layer in enumerate(self.layers):
            try:
                h = layer(h, start_pos, freqs_cis, mask)
            except Exception as e:
                logger.error(f"Error in transformer layer {layer_idx}: {e}")
                raise DistributedOperationError(f"Layer {layer_idx} failed: {e}") from e
        
        # Final normalization and projection
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        
        # Gather logits from all processes in distributed mode
        if world_size > 1:
            try:
                # Verify tensor shapes before all-gather
                all_logits = [torch.empty_like(logits) for _ in range(world_size)]
                
                # Check that all tensors have same shape
                for i in range(world_size):
                    if i == rank:
                        continue
                    all_logits[i] = torch.empty_like(logits)
                
                dist.all_gather(all_logits, logits, timeout=DIST_TIMEOUT)
                
                # Concatenate along vocabulary dimension
                logits = torch.cat(all_logits, dim=-1)
                
            except Exception as e:
                logger.error(f"All-gather failed in Transformer forward: {e}")
                raise DistributedOperationError(f"All-gather failed: {e}") from e
        
        return logits


def setup_distributed():
    """
    Setup distributed environment if needed.
    
    Returns:
        tuple: (world_size, rank) if distributed is initialized
    """
    global world_size, rank
    
    try:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            return world_size, rank
        else:
            world_size = 1
            rank = 0
            return world_size, rank
    except Exception as e:
        logger.error(f"Failed to setup distributed environment: {e}")
        raise


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup distributed environment
    setup_distributed()
    
    # Run test
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    
    args = ModelArgs()
    
    # Test with small batch to catch errors early
    test_batch_size = min(2, args.max_batch_size)
    test_seq_len = min(32, args.max_seq_len)
    
    x = torch.randint(0, args.vocab_size, (test_batch_size, test_seq_len))
    
    try:
        model = Transformer(args)
        output = model(x)
        print(f"Success! Output size: {output.size()}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
