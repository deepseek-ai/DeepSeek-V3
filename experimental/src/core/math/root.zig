const std = @import("std");

// Math utilities for DeepSeek V3
pub const simd = @import("simd.zig");
pub const activation = @import("activation.zig");
pub const rms_norm = @import("rms_norm.zig");

// Re-export common math functions
pub const sqrt = std.math.sqrt;
pub const exp = std.math.exp;
pub const tanh = std.math.tanh;
pub const sin = std.math.sin;
pub const cos = std.math.cos; 