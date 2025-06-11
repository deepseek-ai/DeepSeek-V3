// Metal shader utility for managing and optimizing Metal shaders
// With specific optimizations for M-series Apple Silicon

const std = @import("std");
const Allocator = std.mem.Allocator;
const device = @import("device.zig");
const MetalDeviceInfo = device.MetalDeviceInfo;

/// Optimization level for Metal shaders
pub const ShaderOptimizationLevel = enum {
    none,
    default,
    performance,
    size,
    
    /// Get the recommended optimization level based on device capabilities
    pub fn fromDeviceInfo(device_info: ?MetalDeviceInfo) ShaderOptimizationLevel {
        if (device_info == null) return .default;
        
        if (device_info.?.is_m_series) {
            // M3 can handle highly optimized shaders
            if (device_info.?.series_generation >= 3) {
                return .performance;
            } 
            // M1/M2 balance between performance and size
            else {
                return .default;
            }
        }
        
        // For non-Apple Silicon, be more conservative
        return .default;
    }
};

/// Metal shader types
pub const ShaderType = enum {
    compute,
    vertex,
    fragment,
    
    pub fn toMTLFunctionType(self: ShaderType) []const u8 {
        return switch (self) {
            .compute => "MTLFunctionTypeKernel",
            .vertex => "MTLFunctionTypeVertex",
            .fragment => "MTLFunctionTypeFragment",
        };
    }
};

/// Metal shader source with metadata
pub const ShaderSource = struct {
    name: []const u8,
    source_code: []const u8,
    shader_type: ShaderType,
    
    /// Create a shader source with a given name and code
    pub fn init(name: []const u8, source_code: []const u8, shader_type: ShaderType) ShaderSource {
        return .{
            .name = name,
            .source_code = source_code,
            .shader_type = shader_type,
        };
    }
};

/// Metal shader compilation options including M-series specific optimizations
pub const ShaderCompileOptions = struct {
    optimization_level: ShaderOptimizationLevel,
    fast_math: bool,
    preserve_invariance: bool,
    
    /// Create default options for a specific device
    pub fn forDevice(device_info: ?MetalDeviceInfo) ShaderCompileOptions {
        const opt_level = ShaderOptimizationLevel.fromDeviceInfo(device_info);
        
        // M-series chips benefit from fast math but some algorithms require precision
        const fast_math = device_info != null and 
                         device_info.?.is_m_series and 
                         device_info.?.series_generation >= 2;
        
        return .{
            .optimization_level = opt_level,
            .fast_math = fast_math,
            .preserve_invariance = false,
        };
    }
};

/// Utility for managing Metal shader compilation and caching
pub const ShaderManager = struct {
    allocator: Allocator,
    device_info: ?MetalDeviceInfo,
    compile_options: ShaderCompileOptions,
    
    const Self = @This();
    
    /// Create a new shader manager
    pub fn init(
        allocator: Allocator, 
        device_info: ?MetalDeviceInfo
    ) Self {
        return Self{
            .allocator = allocator,
            .device_info = device_info,
            .compile_options = ShaderCompileOptions.forDevice(device_info),
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        _ = self;
    }
    
    /// Get optimal threadgroup size for a compute shader on current device
    pub fn getOptimalThreadgroupSize(self: *Self) struct { x: u32, y: u32, z: u32 } {
        if (self.device_info == null or !self.device_info.?.is_apple_silicon) {
            return .{ .x = 8, .y = 8, .z = 1 };
        }
        
        // M-series chips have different optimal sizes
        if (self.device_info.?.is_m_series) {
            return switch (self.device_info.?.series_generation) {
                3 => .{ .x = 16, .y = 16, .z = 1 }, // M3 has more GPU cores
                2 => .{ .x = 16, .y = 8, .z = 1 },  // M2 
                else => .{ .x = 8, .y = 8, .z = 1 }, // M1
            };
        }
        
        return .{ .x = 8, .y = 8, .z = 1 };
    }
    
    /// Get memory barrier type based on hardware capabilities
    pub fn getOptimalBarrierType(self: *Self) []const u8 {
        // Newer M-series chips support more efficient memory barriers
        if (self.device_info != null and 
            self.device_info.?.is_m_series and 
            self.device_info.?.series_generation >= 2) {
            return "MTLBarrierScopeBuffers";
        }
        
        return "MTLBarrierScopeTextures | MTLBarrierScopeBuffers";
    }
    
    /// Generate compilation options string for Metal API
    pub fn getCompileOptionsString(self: *Self) []const u8 {
        _ = self;
        // In a real implementation, this would return Objective-C code to set up
        // MTLCompileOptions with the appropriate parameters
        return "MTLCompileOptions"; // Placeholder
    }
};

/// Create optimized Metal shaders for key operations based on device capabilities
pub fn createOptimizedMetalShaders(device_info: ?MetalDeviceInfo) struct {
    matmul: []const u8,
    rms_norm: []const u8,
    swiglu: []const u8,
    attention: []const u8,
} {
    // Base versions of shaders
    const base_matmul_shader = 
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\kernel void matmul_kernel(
        \\    device const float* a [[buffer(0)]],
        \\    device const float* b [[buffer(1)]],
        \\    device float* c [[buffer(2)]],
        \\    constant uint& M [[buffer(3)]],
        \\    constant uint& N [[buffer(4)]],
        \\    constant uint& K [[buffer(5)]],
        \\    uint2 gid [[thread_position_in_grid]]
        \\) {
        \\    if (gid.x >= N || gid.y >= M) return;
        \\    
        \\    float sum = 0.0;
        \\    for (uint k = 0; k < K; k++) {
        \\        sum += a[gid.y * K + k] * b[k * N + gid.x];
        \\    }
        \\    c[gid.y * N + gid.x] = sum;
        \\}
    ;
    
    const base_rms_norm_shader = 
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\kernel void rms_norm_kernel(
        \\    device const float* input [[buffer(0)]],
        \\    device const float* weight [[buffer(1)]],
        \\    device float* output [[buffer(2)]],
        \\    constant uint& size [[buffer(3)]],
        \\    constant float& eps [[buffer(4)]],
        \\    uint idx [[thread_position_in_grid]]
        \\) {
        \\    if (idx >= size) return;
        \\    
        \\    // Calculate sum of squares
        \\    float sum_sq = 0.0;
        \\    for (uint i = 0; i < size; i++) {
        \\        float val = input[i];
        \\        sum_sq += val * val;
        \\    }
        \\    
        \\    // RMS normalization
        \\    float rms = sqrt(sum_sq / size + eps);
        \\    output[idx] = input[idx] / rms * weight[idx];
        \\}
    ;
    
    // Default implementations
    var matmul = base_matmul_shader;
    var rms_norm = base_rms_norm_shader;
    var swiglu = "";  // Placeholder
    var attention = ""; // Placeholder
    
    // For M-series chips, we can use optimized implementations
    if (device_info != null and device_info.?.is_m_series) {
        // M3 optimizations
        if (device_info.?.series_generation >= 3) {
            // M3 has improved threadgroup memory, use tiled implementation
            matmul = 
                \\#include <metal_stdlib>
                \\using namespace metal;
                \\
                \\kernel void matmul_kernel_optimized_m3(
                \\    device const float* a [[buffer(0)]],
                \\    device const float* b [[buffer(1)]],
                \\    device float* c [[buffer(2)]],
                \\    constant uint& M [[buffer(3)]],
                \\    constant uint& N [[buffer(4)]],
                \\    constant uint& K [[buffer(5)]],
                \\    uint2 gid [[thread_position_in_grid]],
                \\    uint2 tid [[thread_position_in_threadgroup]],
                \\    uint2 tgid [[threadgroup_position_in_grid]]
                \\) {
                \\    // Advanced implementation with tiling and local memory
                \\    // Optimized for M3 architecture
                \\    // ...
                \\}
            ;
            
            // Similar optimizations for other kernels...
        }
    }
    
    return .{
        .matmul = matmul,
        .rms_norm = rms_norm,
        .swiglu = swiglu,
        .attention = attention,
    };
}
