const std = @import("std");
const Allocator = std.mem.Allocator;

/// Backend types supported by DeepSeek V3
pub const BackendType = enum {
    cpu,
    metal,
    cuda,
    webgpu,
};

/// Backend capabilities
pub const Capabilities = struct {
    supports_fp16: bool,
    supports_bf16: bool,
    supports_int8: bool,
    max_memory_gb: u32,
    compute_capability: ?[]const u8, // For CUDA
    simd_width: u32,
};

/// Backend interface for different compute backends
pub const Backend = struct {
    type: BackendType,
    device_id: u32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, backend_type: BackendType, device_id: u32) Self {
        return Self{
            .type = backend_type,
            .device_id = device_id,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        // TODO: Backend-specific cleanup
        _ = self;
    }

    pub fn capabilities(self: *const Self) Capabilities {
        return switch (self.type) {
            .cpu => Capabilities{
                .supports_fp16 = true,
                .supports_bf16 = true,
                .supports_int8 = true,
                .max_memory_gb = 128, // Typical system RAM
                .compute_capability = null,
                .simd_width = if (@import("builtin").cpu.arch == .x86_64) 8 else 4,
            },
            .metal => Capabilities{
                .supports_fp16 = true,
                .supports_bf16 = true,
                .supports_int8 = true,
                .max_memory_gb = 96, // Apple Silicon unified memory
                .compute_capability = null,
                .simd_width = 16, // Metal SIMD groups
            },
            .cuda => Capabilities{
                .supports_fp16 = true,
                .supports_bf16 = true,
                .supports_int8 = true,
                .max_memory_gb = 80, // H100 VRAM
                .compute_capability = "8.0", // TODO: Detect actual capability
                .simd_width = 32, // CUDA warp size
            },
            .webgpu => Capabilities{
                .supports_fp16 = false, // Limited support
                .supports_bf16 = false,
                .supports_int8 = false,
                .max_memory_gb = 4, // Browser limitations
                .compute_capability = null,
                .simd_width = 1,
            },
        };
    }

    pub fn name(self: *const Self) []const u8 {
        return switch (self.type) {
            .cpu => "CPU",
            .metal => "Metal",
            .cuda => "CUDA",
            .webgpu => "WebGPU",
        };
    }
};
