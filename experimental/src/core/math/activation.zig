const std = @import("std");

/// SwiGLU activation function used in DeepSeek V3
pub fn swiglu(x: f32, gate: f32) f32 {
    return x * swish(gate);
}

/// Swish activation (SiLU)
pub fn swish(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// GELU activation
pub fn gelu(x: f32) f32 {
    const tanh_arg = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(tanh_arg));
}

/// ReLU activation
pub fn relu(x: f32) f32 {
    return @max(0.0, x);
}

/// Vectorized SwiGLU for SIMD
pub fn swigluVec(comptime size: comptime_int, x: @Vector(size, f32), gate: @Vector(size, f32)) @Vector(size, f32) {
    return x * swishVec(size, gate);
}

/// Vectorized Swish for SIMD
pub fn swishVec(comptime size: comptime_int, x: @Vector(size, f32)) @Vector(size, f32) {
    const ones: @Vector(size, f32) = @splat(1.0);
    return x / (ones + @exp(-x));
} 