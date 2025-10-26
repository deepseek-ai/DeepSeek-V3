from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


def convert_scale_to_ue8m0(scale_fp32: torch.Tensor) -> torch.Tensor:
    """
    Converts a float32 scale tensor to UE8M0 format (uint8 exponent).
    
    Args:
        scale_fp32 (torch.Tensor): Scale tensor in float32 format.
    
    Returns:
        torch.Tensor: Scale tensor in uint8 format (exponent + 127).
    """
    scale_fp32_clamped = torch.clamp(scale_fp32, min=1e-38)
    exponent = torch.ceil(torch.log2(scale_fp32_clamped))
    exponent_biased = (exponent + 127).to(torch.int32)
    exponent_biased = torch.clamp(exponent_biased, 0, 255)
    return exponent_biased.to(torch.uint8)


def convert_scale_from_ue8m0(scale_uint8: torch.Tensor) -> torch.Tensor:
    """
    Converts a UE8M0 format scale tensor (uint8 exponent) to float32.
    
    Args:
        scale_uint8 (torch.Tensor): Scale tensor in uint8 format (exponent + 127).
    
    Returns:
        torch.Tensor: Scale tensor in float32 format (2^exponent).
    """
    exponent = scale_uint8.to(torch.int32) - 127
    scale_fp32 = torch.pow(2.0, exponent.to(torch.float32))
    return scale_fp32


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))
    amax = tl.maximum(amax, 1e-4)
    s = amax / 448.
    if scale_fmt == "ue8m0":
        exp = tl.math.ceil(tl.math.log2(s))
        s = tl.math.exp2(exp)
        exp_int = exp.to(tl.int32) + 127
        tl.store(s_ptr + pid, exp_int.to(s_ptr.dtype.element_ty))
    else:
        tl.store(s_ptr + pid, s)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_dtype = torch.uint8 if scale_fmt == "ue8m0" else torch.float32
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=scale_dtype)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    if scale_fmt == "ue8m0":
        s_uint8 = tl.load(s_ptr + pid_m * n + pid_n)
        exp = s_uint8.to(tl.int32) - 127
        s = tl.math.exp2(exp.to(tl.float32))
    else:
        s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    Dequantizes the input tensor `x` using the provided scaling factors `s`.

    Args:
        x (torch.Tensor): The quantized input tensor.
        s (torch.Tensor): The scaling factors.
        block_size (int, optional): The size of the blocks to be used for dequantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.

    Returns:
        torch.Tensor: The dequantized tensor.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    if scale_fmt is None:
        scale_fmt = "ue8m0" if s.dtype == torch.uint8 else None
    if scale_fmt == "ue8m0" and s.dtype != torch.uint8:
        s = convert_scale_to_ue8m0(s)
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y


fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    scale_fmt: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A (FP8).
        b_ptr (tl.tensor): Pointer to the second input matrix B (FP8).
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
                            (uint8 if scale_fmt=="ue8m0", float32 otherwise)
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
                            (uint8 if scale_fmt=="ue8m0", float32 otherwise)
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        scale_fmt (tl.constexpr): Scale format ("ue8m0" for uint8 exponent format, None for float32).

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        
        if scale_fmt == "ue8m0":
            a_s_uint8 = tl.load(a_s_ptrs)
            b_s_uint8 = tl.load(b_s_ptrs)
            a_exp = a_s_uint8.to(tl.int32) - 127
            b_exp = b_s_uint8.to(tl.int32) - 127
            combined_exp = a_exp[:, None] + b_exp[None, :]
            scale = tl.math.exp2(combined_exp.to(tl.float32))
            accumulator += tl.dot(a, b) * scale
        else:
            a_s = tl.load(a_s_ptrs)
            b_s = tl.load(b_s_ptrs)
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor, scale_fmt: Optional[str] = None):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    if scale_fmt is None:
        if a_s.dtype == torch.uint8 or b_s.dtype == torch.uint8:
            scale_fmt = "ue8m0"
        else:
            scale_fmt = None
    if scale_fmt == "ue8m0":
        if a_s.dtype != torch.uint8:
            a_s = convert_scale_to_ue8m0(a_s)
        if b_s.dtype != torch.uint8:
            b_s = convert_scale_to_ue8m0(b_s)
    
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K, scale_fmt=scale_fmt)
    return c
