from typing import Tuple
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton import Config


@dataclass
class BlockConfig:
    """Configuration for block sizes in tensor operations."""
    size: int = 128
    size_m: int = 64
    size_n: int = 64
    size_k: int = 128


class QuantizationKernels:
    """Collection of Triton kernels for quantization operations."""
    
    @staticmethod
    @triton.jit
    def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
        """
        Quantizes activation values using block-wise scaling.
        
        Args:
            x_ptr: Input tensor pointer
            y_ptr: Output quantized tensor pointer
            s_ptr: Output scaling factors pointer
            BLOCK_SIZE: Size of processing block
        """
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        s = tl.max(tl.abs(x)) / 448.
        y = x / s
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + pid, s)

    @staticmethod
    @triton.jit
    def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
        """
        Dequantizes weights using block-wise scaling.
        
        Args:
            x_ptr: Quantized weights pointer
            s_ptr: Scaling factors pointer
            y_ptr: Output dequantized tensor pointer
            M: Number of rows
            N: Number of columns
            BLOCK_SIZE: Size of processing block
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        n = tl.cdiv(N, BLOCK_SIZE)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        s = tl.load(s_ptr + pid_m * n + pid_n)
        y = x * s
        tl.store(y_ptr + offs, y, mask=mask)


class MatrixMultKernels:
    """Collection of Triton kernels for matrix multiplication operations."""
    
    @staticmethod
    def get_configs():
        """Generate configurations for FP8 GEMM autotuning."""
        return [
            Config({
                'BLOCK_SIZE_M': block_m,
                'BLOCK_SIZE_N': block_n,
                'BLOCK_SIZE_K': 128
            }, num_stages=num_stages, num_warps=8)
            for block_m in [16, 32, 64]
            for block_n in [32, 64, 128]
            for num_stages in [3, 4, 5, 6]
        ]

    @staticmethod
    @triton.autotune(configs=get_configs(), key=['N', 'K'])
    @triton.jit
    def fp8_gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        a_s_ptr, b_s_ptr,
        M, N: tl.constexpr, K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr
    ):
        """
        Performs FP8 matrix multiplication with scaling factors.
        
        Args:
            a_ptr: First input matrix pointer
            b_ptr: Second input matrix pointer
            c_ptr: Output matrix pointer
            a_s_ptr: First matrix scaling factors pointer
            b_s_ptr: Second matrix scaling factors pointer
            M: First matrix rows
            N: Second matrix columns
            K: Inner dimension
            BLOCK_SIZE_M/N/K: Block sizes for tiling
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        k = tl.cdiv(K, BLOCK_SIZE_K)
        
        # Calculate offsets
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # Initialize pointers
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
        a_s_ptrs = a_s_ptr + offs_m * k
        b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Main computation loop
        for i in range(k):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
            a_s = tl.load(a_s_ptrs)
            b_s = tl.load(b_s_ptrs)
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            
            # Update pointers
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K
            a_s_ptrs += 1
            b_s_ptrs += 1

        # Store results
        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)


class TensorOps:
    """High-level interface for tensor operations."""

    @staticmethod
    def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations using block-wise scaling.
        
        Args:
            x: Input tensor
            block_size: Block size for quantization
            
        Returns:
            Tuple of quantized tensor and scaling factors
        """
        assert x.is_contiguous()
        assert x.size(-1) % block_size == 0
        
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
        
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        QuantizationKernels.act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
        
        return y, s

    @staticmethod
    def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        """
        Dequantize weights using block-wise scaling.
        
        Args:
            x: Quantized weight tensor
            s: Scaling factors tensor
            block_size: Block size for dequantization
            
        Returns:
            Dequantized tensor
        """
        assert x.is_contiguous() and s.is_contiguous()
        assert x.dim() == 2 and s.dim() == 2
        
        M, N = x.size()
        y = torch.empty_like(x, dtype=torch.get_default_dtype())
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE']),
            triton.cdiv(N, meta['BLOCK_SIZE'])
        )
        QuantizationKernels.weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
        
        return y

    @staticmethod
    def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
        """
        Perform FP8 matrix multiplication.
        
        Args:
            a: First input matrix
            a_s: First matrix scaling factors
            b: Second input matrix
            b_s: Second matrix scaling factors
            
        Returns:
            Result matrix
        """
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        
        K = a.size(-1)
        M = a.numel() // K
        N = b.size(0)
        
        c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']),
            triton.cdiv(N, META['BLOCK_SIZE_N'])
        )
        MatrixMultKernels.fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
        
        return c