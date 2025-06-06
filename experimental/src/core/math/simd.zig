const std = @import("std");

/// SIMD utilities for high-performance computation
pub fn vectorAdd(comptime T: type, comptime size: comptime_int, a: @Vector(size, T), b: @Vector(size, T)) @Vector(size, T) {
    return a + b;
}

pub fn vectorMul(comptime T: type, comptime size: comptime_int, a: @Vector(size, T), b: @Vector(size, T)) @Vector(size, T) {
    return a * b;
}

pub fn vectorFma(comptime T: type, comptime size: comptime_int, a: @Vector(size, T), b: @Vector(size, T), c: @Vector(size, T)) @Vector(size, T) {
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