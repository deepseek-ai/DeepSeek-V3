from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


# FP8 E4M3 max representable value
FP8_E4M3_MAX = 448.0


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, n_elements, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        n_elements (int): Total number of elements in the input tensor.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
        scale_fmt (tl.constexpr): Format for quantization scale ("ue8m0" or None).

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Boundary masking for partial blocks at the end
    mask = offs < n_elements

    # Load with masking, using 0.0 for out-of-bounds elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute max absolute value with masked reduction
    amax = tl.max(tl.where(mask, tl.abs(x), 0.0))
    amax = tl.maximum(amax, 1e-4)  # Clamp to avoid division by zero

    # Compute scale factor
    s = amax / FP8_E4M3_MAX
    if scale_fmt == "ue8m0":
        # Round scale to power of 2 for ue8m0 format
        exp = tl.math.ceil(tl.math.log2(s))
        s = tl.math.exp2(exp)

    # Quantize and store
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale ("ue8m0" or None). Default is None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    n_elements = x.numel()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, n_elements, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
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
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# Optimized autotuning configurations for FP8 GEMM
# - Extended block sizes for better performance on larger matrices
# - Dynamic num_warps based on block size for optimal occupancy
# - Refined stage counts based on typical workloads
def _get_num_warps(block_m: int, block_n: int) -> int:
    """Calculate optimal num_warps based on block dimensions."""
    total_threads = block_m * block_n
    if total_threads >= 8192:
        return 16
    elif total_threads >= 4096:
        return 8
    elif total_threads >= 2048:
        return 4
    return 4

fp8_gemm_configs = [
    Config(
        {'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128},
        num_stages=num_stages,
        num_warps=_get_num_warps(block_m, block_n)
    )
    for block_m in [16, 32, 64, 128]
    for block_n in [32, 64, 128, 256]
    for num_stages in [3, 4, 5]
    if block_m * block_n <= 16384  # Limit to reasonable tile sizes
]

@triton.autotune(configs=fp8_gemm_configs, key=['M', 'N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    This kernel computes C = A @ B.T where A and B are FP8 quantized matrices
    with per-block scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A (M x K).
        b_ptr (tl.tensor): Pointer to the second input matrix B (N x K, stored transposed).
        c_ptr (tl.tensor): Pointer to the output matrix C (M x N).
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        stride_am, stride_ak: Strides for matrix A.
        stride_bk, stride_bn: Strides for matrix B.
        stride_cm, stride_cn: Strides for matrix C.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Number of K blocks
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

    # Compute block offsets with boundary handling
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Use modulo for wraparound to avoid boundary checks in inner loop
    offs_m_wrap = offs_m % M
    offs_n_wrap = offs_n % N

    # Initialize pointers with proper strides
    a_ptrs = a_ptr + offs_m_wrap[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n_wrap[None, :] * stride_bn + offs_k[:, None] * stride_bk

    # Scale pointers
    a_s_ptrs = a_s_ptr + offs_m_wrap * num_k_blocks
    b_s_ptrs = b_s_ptr + (offs_n_wrap // BLOCK_SIZE_K) * num_k_blocks

    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_block in range(num_k_blocks):
        # Compute remaining K elements for masking
        k_remaining = K - k_block * BLOCK_SIZE_K

        # Load A and B tiles with masking
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        # Load scale factors
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)

        # Compute scaled dot product
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        a_s_ptrs += 1
        b_s_ptrs += 1

    # Convert accumulator to output dtype
    c = accumulator.to(c_ptr.dtype.element_ty)

    # Store output with boundary masking
    offs_m_out = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_out = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m_out[:, None] * stride_cm + offs_n_out[None, :] * stride_cn
    mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Computes C = A @ B.T where A is (M, K) and B is (N, K) stored in transposed layout.

    Args:
        a (torch.Tensor): The first input matrix of shape (..., K), must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix of shape (N, K), must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication with shape (..., N).
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)

    # Create output tensor
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    # Reshape a for 2D GEMM computation
    a_2d = a.view(M, K)
    c_2d = c.view(M, N)

    # Get strides
    stride_am, stride_ak = a_2d.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_cn = c_2d.stride()

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](
        a_2d, b, c_2d,
        a_s, b_s,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return c
