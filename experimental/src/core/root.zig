// DeepSeek V3 Core Module
// This module contains the fundamental components for LLM inference

const std = @import("std");

// Core components
pub const Tensor = @import("tensor.zig").Tensor;
pub const Model = @import("model.zig").Model;
pub const Transformer = @import("transformer.zig").Transformer;
pub const Attention = @import("attention.zig").Attention;
pub const MoE = @import("moe.zig").MoE;
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const Backend = @import("backend.zig").Backend;

// Math utilities
pub const math = @import("math/root.zig");

// Memory management
pub const memory = @import("memory.zig");

// Configuration
pub const Config = @import("config.zig").Config;

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