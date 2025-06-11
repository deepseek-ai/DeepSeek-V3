// Test program for M series detection
const std = @import("std");
const metal_device = @import("backends/metal/device.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    
    const allocator = gpa.allocator();
    
    std.log.info("Testing M series detection...", .{});
    
    // Detect Apple Silicon and M-series capabilities
    const device_info = try metal_device.detectAppleSilicon(allocator);
    defer {
        allocator.free(device_info.device_name);
        allocator.free(device_info.variant);
    }
    
    std.log.info("Device Info:", .{});
    std.log.info("  Device Name: {s}", .{device_info.device_name});
    std.log.info("  Is Apple Silicon: {}", .{device_info.is_apple_silicon});
    std.log.info("  Is M Series: {}", .{device_info.is_m_series});
    
    if (device_info.is_m_series) {
        std.log.info("  M Series Generation: {}", .{device_info.series_generation});
        std.log.info("  Variant: {s}", .{device_info.variant});
    }
    
    std.log.info("  Unified Memory: {} GB", .{device_info.unified_memory_size / (1024 * 1024 * 1024)});
    std.log.info("  Has Apple Neural Engine: {}", .{device_info.has_anc});
    
    // Test other utility functions
    std.log.info("Optimal Work Group Size: {}", .{metal_device.getOptimalWorkGroupSize()});
    std.log.info("Memory Strategy: {s}", .{@tagName(metal_device.getMemoryStrategy())});
    std.log.info("Optimal Tensor Block Size: {}", .{metal_device.getOptimalTensorBlockSize()});
    
    std.log.info("Test complete!", .{});
}
