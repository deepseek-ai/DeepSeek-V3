import torch
import triton
import triton.language as tl


@triton.jit
def weight_dequant_kernel(
    q_ptr, s_ptr, out_ptr, M, N,
    stride_qm, stride_qn, stride_sm, stride_sn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """
    Triton kernel for FP8 weight dequantization.
    out = q * s
    """
    pid = tl.program_id(axis=0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn
    s_ptrs = s_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0)
    s = tl.load(s_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=1)

    out = q.to(tl.float32) * s.to(tl.float32)
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Triton kernel for FP8 GEMM (General Matrix Multiply)
    c = a @ b
    """
    pid = tl.program_id(axis=0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)

        acc += tl.dot(a, b)

    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def dequantize_weights(q_weight: torch.Tensor, scale: torch.Tensor, block_size=16) -> torch.Tensor:
    """
    Dequantizes FP8 weights using provided scaling factors.

    Args:
        q_weight (torch.Tensor): Quantized weight matrix (e.g. float8).
        scale (torch.Tensor): Scaling factors.
        block_size (int): Block size used in the kernel.

    Returns:
        torch.Tensor: Dequantized weight matrix (float32).
    """
    assert q_weight.shape == scale.shape, "Mismatched shapes between quantized weights and scales."

    M, N = q_weight.shape
    output = torch.empty_like(q_weight, dtype=torch.float32)

    grid = (triton.cdiv(M, block_size) * triton.cdiv(N, block_size),)
    weight_dequant_kernel[grid](
        q_weight, scale, output,
        M, N,
        q_weight.stride(0), q_weight.stride(1),
        scale.stride(0), scale.stride(1),
        output.stride(0), output.stride(1),
        block_size, block_size
    )
    return output


def fp8_gemm(a: torch.Tensor, b: torch.Tensor, block_size=16) -> torch.Tensor:
    """
    Performs GEMM on FP8 dequantized matrices using Triton.

    Args:
        a (torch.Tensor): Left matrix (float32).
        b (torch.Tensor): Right matrix (float32).
        block_size (int): Block size for tiling.

    Returns:
        torch.Tensor: Output matrix (float32).
    """
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions."

    M, K = a.shape
    _, N = b.shape
    output = torch.empty((M, N), dtype=torch.float32)

    grid = (triton.cdiv(M, block_size) * triton.cdiv(N, block_size),)
    fp8_gemm_kernel[grid](
        a, b, output,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        block_size, block_size, block_size
    )
    return output
