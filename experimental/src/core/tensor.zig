const std = @import("std");
const Allocator = std.mem.Allocator;
const CoreError = @import("root.zig").CoreError;

pub const TensorError = CoreError || error{
    ShapeMismatch,
    InvalidDimension,
    BufferTooSmall,
};

/// Shape of a tensor - maximum 8 dimensions for DeepSeek V3
pub const Shape = struct {
    dims: [8]u32,
    ndim: u8,
    
    pub fn init(dimensions: []const u32) Shape {
        var shape = Shape{
            .dims = [_]u32{0} ** 8,
            .ndim = @intCast(dimensions.len),
        };
        for (dimensions, 0..) |dim, i| {
            shape.dims[i] = dim;
        }
        return shape;
    }
    
    pub fn numel(self: Shape) u64 {
        var total: u64 = 1;
        for (0..self.ndim) |i| {
            total *= self.dims[i];
        }
        return total;
    }
    
    pub fn equals(self: Shape, other: Shape) bool {
        if (self.ndim != other.ndim) return false;
        for (0..self.ndim) |i| {
            if (self.dims[i] != other.dims[i]) return false;
        }
        return true;
    }
    
    pub fn format(
        self: Shape,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Shape([");
        for (0..self.ndim) |i| {
            if (i > 0) try writer.print(", ");
            try writer.print("{}", .{self.dims[i]});
        }
        try writer.print("])");
    }
};

/// Tensor data type
pub const DType = enum {
    f32,
    f16,
    bf16,
    i32,
    u32,
    i8,
    u8,
    
    pub fn size(self: DType) u8 {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .f16, .bf16 => 2,
            .i8, .u8 => 1,
        };
    }
};

/// Multi-dimensional tensor with SIMD optimizations
pub const Tensor = struct {
    data: []u8,
    shape: Shape,
    dtype: DType,
    allocator: Allocator,
    
    const Self = @This();
    
    /// Create a new tensor with given shape and data type
    pub fn init(allocator: Allocator, shape: Shape, dtype: DType) !Self {
        const size = shape.numel() * dtype.size();
        const data = try allocator.alloc(u8, size);
        @memset(data, 0);
        
        return Self{
            .data = data,
            .shape = shape,
            .dtype = dtype,
            .allocator = allocator,
        };
    }
    
    /// Create tensor from existing data (takes ownership)
    pub fn fromData(allocator: Allocator, data: []u8, shape: Shape, dtype: DType) !Self {
        const expected_size = shape.numel() * dtype.size();
        if (data.len != expected_size) {
            return TensorError.BufferTooSmall;
        }
        
        return Self{
            .data = data,
            .shape = shape,
            .dtype = dtype,
            .allocator = allocator,
        };
    }
    
    /// Create tensor filled with zeros
    pub fn zeros(allocator: Allocator, shape: Shape, dtype: DType) !Self {
        return init(allocator, shape, dtype);
    }
    
    /// Create tensor filled with ones
    pub fn ones(allocator: Allocator, shape: Shape, dtype: DType) !Self {
        var tensor = try init(allocator, shape, dtype);
        try tensor.fill(1.0);
        return tensor;
    }
    
    /// Free tensor memory
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
    }
    
    /// Fill tensor with a scalar value
    pub fn fill(self: *Self, value: f32) !void {
        switch (self.dtype) {
            .f32 => {
                const data_f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, self.data)));
                @memset(data_f32, value);
            },
            .f16 => {
                const data_f16 = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, self.data)));
                @memset(data_f16, @floatCast(value));
            },
            .i32 => {
                const data_i32 = @as([]i32, @alignCast(std.mem.bytesAsSlice(i32, self.data)));
                @memset(data_i32, @intFromFloat(value));
            },
            else => return TensorError.UnsupportedOperation,
        }
    }
    
    /// Get tensor as typed slice (f32)
    pub fn asSliceF32(self: *Self) ![]f32 {
        if (self.dtype != .f32) return TensorError.UnsupportedOperation;
        return @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, self.data)));
    }
    
    /// Get tensor as typed slice (f16)
    pub fn asSliceF16(self: *Self) ![]f16 {
        if (self.dtype != .f16) return TensorError.UnsupportedOperation;
        return @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, self.data)));
    }
    
    /// Element-wise addition (SIMD optimized)
    pub fn add(self: *Self, other: *const Self, result: *Self) !void {
        if (!self.shape.equals(other.shape) or !self.shape.equals(result.shape)) {
            return TensorError.ShapeMismatch;
        }
        if (self.dtype != other.dtype or self.dtype != result.dtype) {
            return TensorError.UnsupportedOperation;
        }
        
        switch (self.dtype) {
            .f32 => try addF32SIMD(self.data, other.data, result.data),
            .f16 => try addF16(self.data, other.data, result.data),
            else => return TensorError.UnsupportedOperation,
        }
    }
    
    /// Matrix multiplication (optimized for transformers)
    pub fn matmul(self: *Self, other: *const Self, result: *Self) !void {
        if (self.shape.ndim != 2 or other.shape.ndim != 2 or result.shape.ndim != 2) {
            return TensorError.InvalidDimension;
        }
        
        const m = self.shape.dims[0];
        const k = self.shape.dims[1];
        const n = other.shape.dims[1];
        
        if (other.shape.dims[0] != k or result.shape.dims[0] != m or result.shape.dims[1] != n) {
            return TensorError.ShapeMismatch;
        }
        
        switch (self.dtype) {
            .f32 => try matmulF32(self, other, result),
            else => return TensorError.UnsupportedOperation,
        }
    }
    
    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Tensor({}, {})", .{ self.shape, @tagName(self.dtype) });
    }
};

// SIMD optimized addition for f32
fn addF32SIMD(a: []const u8, b: []const u8, result: []u8) !void {
    const a_f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, a)));
    const b_f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, b)));
    const result_f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, result)));
    
    const VecSize = 8; // AVX2 can process 8 f32s at once
    const vec_len = a_f32.len / VecSize * VecSize;
    
    // SIMD loop
    var i: usize = 0;
    while (i < vec_len) : (i += VecSize) {
        const va: @Vector(VecSize, f32) = a_f32[i..i+VecSize][0..VecSize].*;
        const vb: @Vector(VecSize, f32) = b_f32[i..i+VecSize][0..VecSize].*;
        const vr = va + vb;
        result_f32[i..i+VecSize][0..VecSize].* = vr;
    }
    
    // Handle remainder
    while (i < a_f32.len) : (i += 1) {
        result_f32[i] = a_f32[i] + b_f32[i];
    }
}

// Basic f16 addition (can be optimized with ARM NEON)
fn addF16(a: []const u8, b: []const u8, result: []u8) !void {
    const a_f16 = @as([]const f16, @alignCast(std.mem.bytesAsSlice(f16, a)));
    const b_f16 = @as([]const f16, @alignCast(std.mem.bytesAsSlice(f16, b)));
    const result_f16 = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, result)));
    
    for (0..a_f16.len) |i| {
        result_f16[i] = a_f16[i] + b_f16[i];
    }
}

// Optimized matrix multiplication for transformers
fn matmulF32(a: *Tensor, b: *const Tensor, c: *Tensor) !void {
    const a_data = try a.asSliceF32();
    const b_data = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, b.data)));
    const c_data = try c.asSliceF32();
    
    const m = a.shape.dims[0];
    const k = a.shape.dims[1];
    const n = b.shape.dims[1];
    
    // TODO: Implement blocked matrix multiplication with SIMD
    // For now, simple triple loop
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |l| {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
}

// Tests
test "tensor creation and basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    // Test tensor creation
    const shape = Shape.init(&[_]u32{2, 3});
    var tensor = try Tensor.zeros(allocator, shape, .f32);
    defer tensor.deinit();
    
    try testing.expect(tensor.shape.numel() == 6);
    try testing.expect(tensor.dtype == .f32);
    
    // Test fill
    try tensor.fill(5.0);
    const data = try tensor.asSliceF32();
    try testing.expect(data[0] == 5.0);
    try testing.expect(data[5] == 5.0);
}

test "tensor addition" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const shape = Shape.init(&[_]u32{4});
    var a = try Tensor.ones(allocator, shape, .f32);
    defer a.deinit();
    
    var b = try Tensor.ones(allocator, shape, .f32);
    defer b.deinit();
    try b.fill(2.0);
    
    var result = try Tensor.zeros(allocator, shape, .f32);
    defer result.deinit();
    
    try a.add(&b, &result);
    
    const data = try result.asSliceF32();
    for (data) |val| {
        try testing.expect(val == 3.0);
    }
} 