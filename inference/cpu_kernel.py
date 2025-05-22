import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def act_quant_cpu(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-compatible version of act_quant. Quantizes the input tensor using block-wise quantization.
    
    Args:
        x (torch.Tensor): The input tensor to be quantized.
        block_size (int, optional): The size of the blocks for quantization. Default is 128.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scaling factors
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    
    # Handle non-divisible cases more gracefully
    if x.size(-1) % block_size != 0:
        # Pad the tensor to make it divisible by block_size
        pad_size = block_size - (x.size(-1) % block_size)
        x = F.pad(x, (0, pad_size))
    
    # Reshape to blocks for efficient processing
    shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    
    # Calculate scaling factors (max absolute value in each block)
    s = torch.max(torch.abs(x_reshaped), dim=1, keepdim=True)[0] / 448.0
    
    # Avoid division by zero
    s = torch.clamp(s, min=1e-10)
    
    # Quantize by dividing by scaling factors
    y = x_reshaped / s
    
    # Either use float8 if available or simulate with int8 + scaling
    if hasattr(torch, "float8_e4m3fn"):
        y = y.to(torch.float8_e4m3fn)
    else:
        # Simulate float8 with int8 quantization
        y = torch.clamp(y, -448.0, 448.0)
        y = (y / 448.0 * 127).round().to(torch.int8)
    
    # Reshape back to original shape
    y = y.reshape(shape)
    
    # Reshape scaling factors to match expected output format
    s = s.reshape(*shape[:-1], -1).squeeze(-1)
    
    return y, s

def weight_dequant_cpu(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    CPU-compatible version of weight_dequant. Dequantizes the weight tensor.
    
    Args:
        weight (torch.Tensor): Quantized weight tensor.
        scale (torch.Tensor): Scaling factors.
        
    Returns:
        torch.Tensor: Dequantized weight tensor.
    """
    # Handle different quantization formats
    if weight.dtype == torch.int8:
        # For int8 simulated quantization
        weight_float = weight.to(torch.float32) / 127.0 * 448.0
    elif hasattr(torch, "float8_e4m3fn") and weight.dtype == torch.float8_e4m3fn:
        # Native float8 support
        weight_float = weight.to(torch.float32)
    else:
        # Already in a floating point format
        weight_float = weight.to(torch.float32)
    
    # Reshape scale to broadcast correctly
    if weight.dim() == 2:
        # For linear layers
        out_features, in_features = weight.shape
        block_size = 128  # Same as in the original code
        
        scale_out = (out_features + block_size - 1) // block_size
        scale_in = (in_features + block_size - 1) // block_size
        
        if scale.numel() == scale_out * scale_in:
            # Reshape to match weight blocks
            scale_reshaped = scale.reshape(scale_out, scale_in)
            
            # Create a mask for each block
            out_blocks = torch.arange(out_features).reshape(-1, 1) // block_size
            in_blocks = torch.arange(in_features).reshape(1, -1) // block_size
            
            # Limit to actual dimensions
            out_blocks = torch.clamp(out_blocks, max=scale_out-1)
            in_blocks = torch.clamp(in_blocks, max=scale_in-1)
            
            # Get corresponding scale for each position
            scale_broadcast = scale_reshaped[out_blocks, in_blocks]
            
            # Apply scaling
            return weight_float * scale_broadcast
    
    # Fallback for other tensor dimensions
    return weight_float * scale

def fp8_gemm_cpu(x: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """
    CPU-compatible version of fp8_gemm. Performs matrix multiplication with quantized tensors.
    
    Args:
        x (torch.Tensor): Input activations (quantized).
        x_scale (torch.Tensor): Scaling factors for input activations.
        weight (torch.Tensor): Weights (quantized).
        weight_scale (torch.Tensor): Scaling factors for weights.
        
    Returns:
        torch.Tensor: Result of matrix multiplication.
    """
    # Dequantize input and weights
    if x.dtype == torch.int8:
        x_float = x.to(torch.float32) / 127.0 * 448.0
    elif hasattr(torch, "float8_e4m3fn") and x.dtype == torch.float8_e4m3fn:
        x_float = x.to(torch.float32)
    else:
        x_float = x
    
    # Apply input scaling
    if x_scale is not None:
        # Reshape x_scale for broadcasting
        new_shape = list(x_scale.shape) + [1] * (x_float.dim() - x_scale.dim())
        x_float = x_float * x_scale.reshape(*new_shape)
    
    # Dequantize weights
    weight_dequant = weight_dequant_cpu(weight, weight_scale)
    
    # Perform matrix multiplication
    result = F.linear(x_float, weight_dequant)
    
    return result

# MPS (Metal Performance Shaders) optimized versions for Apple Silicon
def setup_mps_kernels():
    """
    Set up optimized MPS kernels if running on Apple Silicon
    """
    if hasattr(torch, "mps") and torch.mps.is_available():
        print("Setting up MPS optimized kernels for Apple Silicon")
        # MPS already optimizes most operations automatically
        # Additional optimizations could be added in the future
    else:
        print("MPS not available, using CPU kernels")

# Provide unified interface that selects the appropriate implementation
def get_optimized_kernels(device="cpu"):
    """
    Returns optimized kernel functions based on the device
    
    Args:
        device (str): The device to optimize for ("cpu", "mps", or "cuda")
        
    Returns:
        dict: Dictionary of optimized kernel functions
    """
    if device == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
        setup_mps_kernels()
        # For MPS, we use the CPU implementations which will be automatically 
        # optimized by PyTorch's MPS backend
        return {
            "act_quant": act_quant_cpu,
            "weight_dequant": weight_dequant_cpu,
            "fp8_gemm": fp8_gemm_cpu
        }
    elif device == "cuda" and torch.cuda.is_available():
        # For CUDA, use the original implementations
        from kernel import act_quant, weight_dequant, fp8_gemm
        return {
            "act_quant": act_quant,
            "weight_dequant": weight_dequant,
            "fp8_gemm": fp8_gemm
        }
    else:
        # Default to CPU implementations
        return {
            "act_quant": act_quant_cpu,
            "weight_dequant": weight_dequant_cpu,
            "fp8_gemm": fp8_gemm_cpu
        }
