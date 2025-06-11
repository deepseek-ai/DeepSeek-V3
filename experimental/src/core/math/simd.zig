const std = @import("std");

/// SIMD utilities for high-performance computation

/// Vector operations for @Vector types
pub fn vecAdd(comptime T: type, comptime size: comptime_int, a: @Vector(size, T), b: @Vector(size, T)) @Vector(size, T) {
    return a + b;
}

pub fn vecMul(comptime T: type, comptime size: comptime_int, a: @Vector(size, T), b: @Vector(size, T)) @Vector(size, T) {
    return a * b;
}

pub fn vecFma(comptime T: type, comptime size: comptime_int, a: @Vector(size, T), b: @Vector(size, T), c: @Vector(size, T)) @Vector(size, T) {
    return @mulAdd(@Vector(size, T), a, b, c);
}

/// Horizontal sum of vector elements
pub fn horizontalSum(comptime T: type, comptime size: comptime_int, vec: @Vector(size, T)) T {
    if (size == 1) return vec[0];
    
    var result: T = 0;
    for (0..size) |i| {
        result += vec[i];
    }
    return result;
}

/// Slice-based SIMD operations for tensor operations
/// Element-wise addition of two slices with SIMD optimization
pub fn vectorAdd(comptime T: type, a: []const T, b: []const T, result: []T) void {
    if (a.len != b.len or a.len != result.len) {
        @panic("SIMD vectorAdd: slice lengths must match");
    }
    
    const len = a.len;
    const vector_size = 4; // Process 4 elements at once
    
    // SIMD processing for bulk of data
    const simd_len = len - (len % vector_size);
    var i: usize = 0;
    while (i < simd_len) : (i += vector_size) {
        const va: @Vector(vector_size, T) = a[i..i+vector_size][0..vector_size].*;
        const vb: @Vector(vector_size, T) = b[i..i+vector_size][0..vector_size].*;
        const vr = va + vb;
        result[i..i+vector_size][0..vector_size].* = vr;
    }
    
    // Handle remaining elements
    while (i < len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// Element-wise multiplication of two slices with SIMD optimization
pub fn vectorMul(comptime T: type, a: []const T, b: []const T, result: []T) void {
    if (a.len != b.len or a.len != result.len) {
        @panic("SIMD vectorMul: slice lengths must match");
    }
    
    const len = a.len;
    const vector_size = 4;
    
    const simd_len = len - (len % vector_size);
    var i: usize = 0;
    while (i < simd_len) : (i += vector_size) {
        const va: @Vector(vector_size, T) = a[i..i+vector_size][0..vector_size].*;
        const vb: @Vector(vector_size, T) = b[i..i+vector_size][0..vector_size].*;
        const vr = va * vb;
        result[i..i+vector_size][0..vector_size].* = vr;
    }
    
    while (i < len) : (i += 1) {
        result[i] = a[i] * b[i];
    }
} 