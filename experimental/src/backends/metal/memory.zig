// Metal-specific memory management for Apple Silicon
// Optimized for the unified memory architecture of M-series chips

const std = @import("std");
const Allocator = std.mem.Allocator;
const device = @import("device.zig");
const MetalDeviceInfo = device.MetalDeviceInfo;

/// Memory modes available for Metal buffers
pub const MetalMemoryMode = enum {
    /// Shared between CPU and GPU with automatic migration
    Shared,
    
    /// Managed with separate CPU and GPU views but synchronized
    Managed,
    
    /// GPU-only storage for maximum performance
    Private,
    
    /// Memory visible to both CPU and GPU (Apple Silicon only)
    Unified,
};

/// Buffer usage patterns to optimize memory allocation
pub const MetalBufferUsage = enum {
    /// Read often by GPU
    GpuRead,
    
    /// Write often by GPU
    GpuWrite,
    
    /// Read/write by both CPU and GPU
    Shared,
    
    /// Used only temporarily for a single operation
    Transient,
};

/// Memory manager for optimal Metal buffer allocation on M-series chips
pub const MetalMemoryManager = struct {
    allocator: Allocator,
    device_info: ?MetalDeviceInfo,
    total_allocated: usize,
    max_allocation: usize,
    
    const Self = @This();
    
    /// Create a new Metal memory manager
    pub fn init(allocator: Allocator, device_info: ?MetalDeviceInfo) Self {
        return Self{
            .allocator = allocator,
            .device_info = device_info,
            .total_allocated = 0,
            .max_allocation = 0,
        };
    }
    
    /// Clean up any resources
    pub fn deinit(self: *Self) void {
        // Release any cached buffers or other resources
        _ = self;
    }
    
    /// Get the optimal memory mode based on device capabilities and usage pattern
    pub fn getOptimalMemoryMode(self: *Self, usage: MetalBufferUsage) MetalMemoryMode {
        // If we're on Apple Silicon, we can use unified memory
        const is_apple_silicon = self.device_info != null and self.device_info.?.is_apple_silicon;
        
        if (is_apple_silicon) {
            return switch (usage) {
                .GpuRead => .Unified,
                .GpuWrite => .Unified,
                .Shared => .Unified,
                .Transient => .Private, // Even on unified memory, transient data is better in private
            };
        } else {
            // On Intel Macs with discrete GPU
            return switch (usage) {
                .GpuRead => .Managed,
                .GpuWrite => .Private,
                .Shared => .Managed,
                .Transient => .Private,
            };
        }
    }
    
    /// Get recommended allocation size (aligned to device preferences)
    pub fn getOptimalAllocationSize(self: *Self, requested_size: usize) usize {
        // M-series chips prefer certain memory alignment patterns
        const alignment: usize = if (self.device_info != null and self.device_info.?.is_m_series) 
            16 * 1024 // 16KB alignment on M-series
        else 
            4 * 1024; // 4KB on other devices
            
        return std.mem.alignForward(usize, requested_size, alignment);
    }
    
    /// Track memory allocations for monitoring
    pub fn trackAllocation(self: *Self, size: usize) void {
        self.total_allocated += size;
        self.max_allocation = std.math.max(self.max_allocation, self.total_allocated);
    }
    
    /// Track memory deallocations
    pub fn trackDeallocation(self: *Self, size: usize) void {
        if (self.total_allocated >= size) {
            self.total_allocated -= size;
        } else {
            self.total_allocated = 0;
        }
    }
    
    /// Get memory usage statistics
    pub fn getMemoryStats(self: *Self) struct { 
        current: usize,
        peak: usize,
        device_total: usize,
    } {
        const device_total = if (self.device_info != null) 
            self.device_info.?.unified_memory_size 
        else 
            0;
            
        return .{
            .current = self.total_allocated,
            .peak = self.max_allocation,
            .device_total = device_total,
        };
    }
    
    /// Get recommended buffer storage mode string for Metal API
    pub fn getStorageModeString(mode: MetalMemoryMode) []const u8 {
        return switch (mode) {
            .Shared => "MTLStorageModeShared",
            .Managed => "MTLStorageModeManaged",
            .Private => "MTLStorageModePrivate",
            .Unified => "MTLStorageModeShared", // Unified uses Shared on the API level
        };
    }
};

/// Helper to determine if hazard tracking should be enabled based on device capabilities
pub fn shouldUseHazardTracking(device_info: ?MetalDeviceInfo) bool {
    if (device_info == null) return false;
    
    // M3 and newer have better hazard tracking hardware
    if (device_info.?.is_m_series and device_info.?.series_generation >= 3) {
        return true;
    }
    
    return false;
}
