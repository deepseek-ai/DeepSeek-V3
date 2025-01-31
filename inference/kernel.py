import torch
import triton
import triton.language as tl

def weight_dequant_kernel(
    q_ptr, s_ptr, out_ptr, M, N, K,
    stride_qm, stride_qk, stride_sm, stride_sn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Kernel para desquantização de pesos FP8.
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
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
    Kernel para multiplicação de matrizes com FP8.
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0)
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

def dequantize_weights(q_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Função para desquantizar pesos FP8 com segurança.
    """
    assert q_weight.shape == scale.shape, "Dimensões incompatíveis entre peso quantizado e escala."
    
    out = torch.empty_like(q_weight, dtype=torch.float32)
    weight_dequant_kernel[
        (q_weight.shape[0] // 16, q_weight.shape[1] // 16)
    ](
        q_weight, scale, out,
        q_weight.shape[0], q_weight.shape[1], q_weight.shape[1],
        q_weight.stride(0), q_weight.stride(1),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        16, 16, 16
    )
    return out

def fp8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiplicação de matrizes FP8 segura e eficiente.
    """
    assert a.shape[1] == b.shape[0], "Dimensões incompatíveis para multiplicação de matrizes."
    
    c = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32)
    fp8_gemm_kernel[
        (a.shape[0] // 16, b.shape[1] // 16)
    ](
        a, b, c,
        a.shape[0], b.shape[1], a.shape[1],
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        16, 16, 16
    )
    return c
