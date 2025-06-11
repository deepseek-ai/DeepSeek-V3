// Metal Device detection and handling for Apple Silicon
// Specifically optimized for M-series chips using proper system detection

const std = @import("std");
const Allocator = std.mem.Allocator;
const c = std.c;

// Device information structure
pub const MetalDeviceInfo = struct {
    device_name: []const u8,
    is_apple_silicon: bool,
    is_m_series: bool,
    series_generation: u8, // 1 = M1, 2 = M2, 3 = M3, etc.
    variant: []const u8, // "Pro", "Max", "Ultra", etc.
    unified_memory_size: u64,
    has_anc: bool, // Apple Neural Engine

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Metal Device: {s} ({s}{d} {s})", .{
            self.device_name,
            if (self.is_m_series) "M" else "",
            if (self.is_m_series) self.series_generation else 0,
            if (self.is_m_series) self.variant else "",
        });
        try writer.print("\nUnified Memory: {} GB", .{self.unified_memory_size / (1024 * 1024 * 1024)});
        try writer.print("\nApple Neural Engine: {}", .{if (self.has_anc) "Available" else "Not Available"});
    }
};

// M-series chip information
const MSeriesInfo = struct {
    is_m_series: bool,
    generation: u8,
    variant: []const u8,
};

// System detection using sysctl
const SysctlError = error{
    NotFound,
    BufferTooSmall,
    SystemError,
};

/// Get sysctl string value
fn getSysctlString(allocator: Allocator, name: []const u8) ![]const u8 {
    // Only available on macOS
    if (@import("builtin").os.tag != .macos) {
        return SysctlError.NotFound;
    }

    var size: usize = 0;
    
    // First, get the size needed
    const name_cstr = try allocator.dupeZ(u8, name);
    defer allocator.free(name_cstr);
    
    if (c.sysctlbyname(name_cstr.ptr, null, &size, null, 0) != 0) {
        return SysctlError.NotFound;
    }
    
    // Allocate buffer and get the actual value
    const buffer = try allocator.alloc(u8, size);
    defer allocator.free(buffer);
    
    if (c.sysctlbyname(name_cstr.ptr, buffer.ptr, &size, null, 0) != 0) {
        return SysctlError.SystemError;
    }
    
    // Return a copy of the string (minus null terminator if present)
    const len = if (size > 0 and buffer[size - 1] == 0) size - 1 else size;
    return try allocator.dupe(u8, buffer[0..len]);
}

/// Get sysctl integer value
fn getSysctlInt(comptime T: type, name: []const u8, allocator: Allocator) !T {
    if (@import("builtin").os.tag != .macos) {
        return SysctlError.NotFound;
    }

    var value: T = 0;
    var size: usize = @sizeOf(T);
    
    const name_cstr = try allocator.dupeZ(u8, name);
    defer allocator.free(name_cstr);
    
    if (c.sysctlbyname(name_cstr.ptr, &value, &size, null, 0) != 0) {
        return SysctlError.NotFound;
    }
    
    return value;
}

/// Check if running under Rosetta 2 translation
fn isRunningUnderRosetta(allocator: Allocator) bool {
    const result = getSysctlInt(i32, "sysctl.proc_translated", allocator) catch return false;
    return result == 1;
}

/// Check if hardware supports ARM64 (Apple Silicon)
fn isAppleSiliconHardware(allocator: Allocator) bool {
    // Check for ARM64 support
    const arm64_support = getSysctlInt(i32, "hw.optional.arm64", allocator) catch return false;
    if (arm64_support == 1) return true;
    
    // Alternative check: CPU architecture
    if (@import("builtin").target.cpu.arch == .aarch64) return true;
    
    // If running under Rosetta, we're on Apple Silicon
    return isRunningUnderRosetta(allocator);
}

/// Parse M-series information from CPU brand string
fn parseMSeriesInfo(cpu_brand: []const u8) MSeriesInfo {
    // Default values
    var result = MSeriesInfo{
        .is_m_series = false,
        .generation = 0,
        .variant = "",
    };
    
    // Look for Apple M pattern
    if (std.mem.indexOf(u8, cpu_brand, "Apple M") == null) {
        return result;
    }
    
    result.is_m_series = true;
    
    // Extract generation and variant from CPU brand string
    // Examples: "Apple M1", "Apple M1 Pro", "Apple M1 Max", "Apple M1 Ultra"
    if (std.mem.indexOf(u8, cpu_brand, "M1")) |_| {
        result.generation = 1;
        if (std.mem.indexOf(u8, cpu_brand, " Pro")) |_| {
            result.variant = "Pro";
        } else if (std.mem.indexOf(u8, cpu_brand, " Max")) |_| {
            result.variant = "Max";
        } else if (std.mem.indexOf(u8, cpu_brand, " Ultra")) |_| {
            result.variant = "Ultra";
        } else {
            // Just "Apple M1" - this is the regular M1
            result.variant = "";
        }
    } else if (std.mem.indexOf(u8, cpu_brand, "M2")) |_| {
        result.generation = 2;
        if (std.mem.indexOf(u8, cpu_brand, " Pro")) |_| {
            result.variant = "Pro";
        } else if (std.mem.indexOf(u8, cpu_brand, " Max")) |_| {
            result.variant = "Max";
        } else if (std.mem.indexOf(u8, cpu_brand, " Ultra")) |_| {
            result.variant = "Ultra";
        } else {
            result.variant = "";
        }
    } else if (std.mem.indexOf(u8, cpu_brand, "M3")) |_| {
        result.generation = 3;
        if (std.mem.indexOf(u8, cpu_brand, " Pro")) |_| {
            result.variant = "Pro";
        } else if (std.mem.indexOf(u8, cpu_brand, " Max")) |_| {
            result.variant = "Max";
        } else if (std.mem.indexOf(u8, cpu_brand, " Ultra")) |_| {
            result.variant = "Ultra";
        } else {
            result.variant = "";
        }
    } else if (std.mem.indexOf(u8, cpu_brand, "M4")) |_| {
        result.generation = 4;
        if (std.mem.indexOf(u8, cpu_brand, " Pro")) |_| {
            result.variant = "Pro";
        } else if (std.mem.indexOf(u8, cpu_brand, " Max")) |_| {
            result.variant = "Max";
        } else if (std.mem.indexOf(u8, cpu_brand, " Ultra")) |_| {
            result.variant = "Ultra";
        } else {
            result.variant = "";
        }
    }
    
    return result;
}

/// Try to detect GPU configuration for more detailed chip identification
fn detectGPUCores(allocator: Allocator) u32 {
    // Try to get GPU core count - this can help distinguish variants
    // Regular M1: 7-8 GPU cores
    // M1 Pro: 14-16 GPU cores  
    // M1 Max: 24-32 GPU cores
    
    // This is a placeholder - actual implementation would query Metal API
    // For now, return 0 to indicate unknown
    _ = allocator;
    return 0;
}

/// Detect Apple Silicon and M-series chip capabilities using proper system detection
pub fn detectAppleSilicon(allocator: Allocator) !MetalDeviceInfo {
    // Check at compile-time if we're on macOS
    const is_macos = @import("builtin").os.tag == .macos;
    if (!is_macos) {
        return MetalDeviceInfo{
            .device_name = try allocator.dupe(u8, "Non-macOS Device"),
            .is_apple_silicon = false,
            .is_m_series = false,
            .series_generation = 0,
            .variant = try allocator.dupe(u8, ""),
            .unified_memory_size = 0,
            .has_anc = false,
        };
    }
    
    // Detect Apple Silicon hardware
    const is_apple_silicon = isAppleSiliconHardware(allocator);
    if (!is_apple_silicon) {
        return MetalDeviceInfo{
            .device_name = try allocator.dupe(u8, "Intel Mac"),
            .is_apple_silicon = false,
            .is_m_series = false,
            .series_generation = 0,
            .variant = try allocator.dupe(u8, ""),
            .unified_memory_size = 0,
            .has_anc = false,
        };
    }
    
    // Get CPU brand string for M-series detection - this is the authoritative source
    const cpu_brand = getSysctlString(allocator, "machdep.cpu.brand_string") catch "Apple Silicon";
    defer allocator.free(cpu_brand);
    
    std.log.debug("CPU Brand String: '{s}'", .{cpu_brand});
    
    // Parse M-series information from the actual CPU brand string
    const m_info = parseMSeriesInfo(cpu_brand);
    
    // Get additional hardware details for logging/debugging
    const hw_model = getSysctlString(allocator, "hw.model") catch "";
    defer if (hw_model.len > 0) allocator.free(hw_model);
    
    const gpu_cores = detectGPUCores(allocator);
    if (gpu_cores > 0) {
        std.log.debug("GPU Cores: {}", .{gpu_cores});
    }
    
    std.log.debug("Hardware Model: '{s}'", .{hw_model});
    std.log.debug("Detected M{d} {s}", .{ m_info.generation, m_info.variant });
    
    // Get system memory
    const memory_size = getSysctlInt(u64, "hw.memsize", allocator) catch (16 * 1024 * 1024 * 1024); // Default 16GB
    
    // Get device name
    const device_name = getSysctlString(allocator, "hw.model") catch "Apple Silicon Mac";
    
    return MetalDeviceInfo{
        .device_name = device_name, // This will be owned by the caller
        .is_apple_silicon = true,
        .is_m_series = m_info.is_m_series,
        .series_generation = m_info.generation,
        .variant = try allocator.dupe(u8, m_info.variant), // Duplicate to ensure consistent allocation
        .unified_memory_size = memory_size,
        .has_anc = m_info.is_m_series, // All M-series have Apple Neural Engine
    };
}

/// Get optimal GPU parameters for detected device
pub fn getOptimalWorkGroupSize() u32 {
    // These are reasonable defaults that should work well on most Apple GPU architectures
    // In a real implementation, we would query Metal API for the actual optimal values
    if (@import("builtin").target.cpu.arch == .aarch64) {
        // Apple Silicon optimized values based on GPU core count
        return 128;
    }
    
    // Default for Intel Macs and others
    return 64;
}

/// Get recommended memory allocation strategy based on device capabilities
pub fn getMemoryStrategy() enum { UnifiedMemory, DiscreteMemory } {
    // Check if we're on Apple Silicon hardware (even under Rosetta)
    if (@import("builtin").os.tag == .macos) {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const allocator = gpa.allocator();
        
        if (isAppleSiliconHardware(allocator)) {
            return .UnifiedMemory; // Apple Silicon uses unified memory
        }
    }
    
    // For Intel Macs and other platforms
    return .DiscreteMemory;
}

/// Get optimal tensor block size for current device
pub fn getOptimalTensorBlockSize() u32 {
    if (@import("builtin").target.cpu.arch == .aarch64) {
        // Apple Silicon has more GPU cores and benefits from larger blocks
        return 256;
    } else {
        return 128;
    }
}
