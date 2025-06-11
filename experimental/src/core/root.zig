// DeepSeek V3 Core Module
// This module contains the fundamental components for LLM inference

const std = @import("std");

pub const Attention = @import("attention.zig").Attention;
pub const Backend = @import("backend.zig").Backend;
pub const blas = @import("blas.zig");
pub const Config = @import("config.zig").Config;
pub const math = @import("math/root.zig");
pub const memory = @import("memory.zig");
pub const Model = @import("model.zig").Model;
pub const MoE = @import("moe.zig").MoE;
pub const Shape = @import("tensor.zig").Shape;
pub const tensor = @import("tensor.zig");
pub const FloatTensor = tensor.FloatTensor;
pub const DoubleTensor = tensor.DoubleTensor;
pub const IntTensor = tensor.IntTensor;
pub const ByteTensor = tensor.ByteTensor;
pub const createMatrix = tensor.createMatrix;
pub const createVector = tensor.createVector;
pub const benchmarkTensorOps = tensor.benchmarkTensorOps;
pub const TensorDType = @import("tensor.zig").TensorDType;
pub const TensorShape = @import("tensor.zig").TensorShape;
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const Transformer = @import("transformer.zig").Transformer;

// Core tensor and math components
// Tensor type aliases for convenience
// Helper functions
// Other core components (may need implementation)
// Math utilities
// Memory management
// Configuration
// Error types
pub const CoreError = error{
    InvalidTensorShape,
    UnsupportedOperation,
    ModelLoadError,
    TokenizerError,
    BackendError,
    OutOfMemory,
    InvalidConfiguration,
};

// Version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;
    pub const string = "0.1.0";
};

// Core test suite
test "core module" {
    const testing = std.testing;

    // Basic smoke tests
    try testing.expect(version.major == 0);
    try testing.expect(version.minor == 1);
}

// Utility functions
pub fn init() void {
    // TODO: Initialize any global state if needed
    std.log.info("DeepSeek V3 Core initialized (v{s})", .{version.string});
}

pub fn deinit() void {
    // TODO: Cleanup any global state
    std.log.info("DeepSeek V3 Core deinitialized");
}
