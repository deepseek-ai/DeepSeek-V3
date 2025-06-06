const std = @import("std");

/// RMS Normalization used in DeepSeek V3
pub fn rmsNorm(input: []const f32, weight: []const f32, output: []f32, eps: f32) void {
    std.debug.assert(input.len == weight.len);
    std.debug.assert(input.len == output.len);
    
    // Compute mean square
    var mean_square: f32 = 0.0;
    for (input) |x| {
        mean_square += x * x;
    }
    mean_square /= @floatFromInt(input.len);
    
    // Compute RMS and normalize
    const rms = @sqrt(mean_square + eps);
    for (0..input.len) |i| {
        output[i] = (input[i] / rms) * weight[i];
    }
}

/// Vectorized RMS normalization for better performance
pub fn rmsNormVec(input: []const f32, weight: []const f32, output: []f32, eps: f32) void {
    const VecSize = 8;
    const vec_len = input.len / VecSize * VecSize;
    
    // Compute mean square using SIMD
    var sum_squares: @Vector(VecSize, f32) = @splat(0.0);
    var i: usize = 0;
    while (i < vec_len) : (i += VecSize) {
        const x: @Vector(VecSize, f32) = input[i..i+VecSize][0..VecSize].*;
        sum_squares += x * x;
    }
    
    // Sum the vector elements
    var mean_square: f32 = 0.0;
    for (0..VecSize) |j| {
        mean_square += sum_squares[j];
    }
    
    // Handle remainder
    while (i < input.len) : (i += 1) {
        mean_square += input[i] * input[i];
    }
    
    mean_square /= @floatFromInt(input.len);
    
    // Normalize using SIMD
    const rms = @sqrt(mean_square + eps);
    const rms_vec: @Vector(VecSize, f32) = @splat(rms);
    
    i = 0;
    while (i < vec_len) : (i += VecSize) {
        const x: @Vector(VecSize, f32) = input[i..i+VecSize][0..VecSize].*;
        const w: @Vector(VecSize, f32) = weight[i..i+VecSize][0..VecSize].*;
        const normalized = (x / rms_vec) * w;
        output[i..i+VecSize][0..VecSize].* = normalized;
    }
    
    // Handle remainder
    while (i < input.len) : (i += 1) {
        output[i] = (input[i] / rms) * weight[i];
    }
} 