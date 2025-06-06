// Benchmark Suite for DeepZig V3 Implementation
// Tests performance of core operations across different backends

const std = @import("std");
const deepseek_core = @import("deepseek_core");
const cpu_backend = @import("cpu_backend");
const print = std.debug.print;

// Import Shape from deepseek_core
const Shape = deepseek_core.Shape;

const BenchmarkResult = struct {
    name: []const u8,
    iterations: u32,
    total_time_ns: u64,
    avg_time_ns: u64,
    ops_per_second: f64,
    memory_used_mb: f64,
    
    pub fn format(
        self: BenchmarkResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "{s:30} | {d:6} iter | {d:8.2} ms | {d:10.0} ops/s | {d:6.1} MB",
            .{ self.name, self.iterations, @as(f64, @floatFromInt(self.avg_time_ns)) / 1_000_000.0, self.ops_per_second, self.memory_used_mb }
        );
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    print("🚀 DeepZig V3 Performance Benchmarks\n", .{});
    print("==========================================\n\n", .{});
    
    // Initialize backends
    var cpu_backend_instance = try cpu_backend.init(allocator);
    defer cpu_backend_instance.deinit();
    
    print("Backend: CPU (SIMD optimized)\n", .{});
    print("Architecture: {s}\n", .{@tagName(@import("builtin").cpu.arch)});
    print("Thread count: {d}\n\n", .{std.Thread.getCpuCount() catch 4});
    
    // Run benchmarks
    var results = std.ArrayList(BenchmarkResult).init(allocator);
    defer results.deinit();
    
    // Tensor operations
    try results.append(try benchmarkTensorCreation(allocator));
    try results.append(try benchmarkTensorAddition(allocator));
    try results.append(try benchmarkMatrixMultiplication(allocator));
    
    // Activation functions
    try results.append(try benchmarkSwiGLU(allocator));
    try results.append(try benchmarkRMSNorm(allocator));
    
    // Memory operations
    try results.append(try benchmarkMemoryBandwidth(allocator));
    
    // Print results
    print("Benchmark Results:\n", .{});
    print("------------------\n", .{});
    print("Operation                      | Iterations |  Avg Time | Operations/s | Memory\n", .{});
    print("-------------------------------|------------|-----------|--------------|-------\n", .{});
    
    for (results.items) |result| {
        print("{}\n", .{result});
    }
    
    print("\n🎯 Benchmark completed!\n", .{});
}

/// Benchmark tensor creation and memory allocation
fn benchmarkTensorCreation(allocator: std.mem.Allocator) !BenchmarkResult {
    const iterations = 1000;
    const shape = Shape.init(&[_]u32{ 1024, 1024 });
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        var tensor = try deepseek_core.Tensor.zeros(allocator, shape, .f32);
        tensor.deinit();
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    const avg_time = total_time / iterations;
    
    return BenchmarkResult{
        .name = "Tensor Creation (1024x1024)",
        .iterations = iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .ops_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0),
        .memory_used_mb = (1024.0 * 1024.0 * 4.0) / (1024.0 * 1024.0), // 4MB tensor
    };
}

/// Benchmark SIMD-optimized tensor addition
fn benchmarkTensorAddition(allocator: std.mem.Allocator) !BenchmarkResult {
    const iterations = 100;
    const shape = Shape.init(&[_]u32{ 4096, 1024 });
    
    var a = try deepseek_core.Tensor.ones(allocator, shape, .f32);
    defer a.deinit();
    
    var b = try deepseek_core.Tensor.ones(allocator, shape, .f32);
    defer b.deinit();
    
    var result = try deepseek_core.Tensor.zeros(allocator, shape, .f32);
    defer result.deinit();
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        try a.add(&b, &result);
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    const avg_time = total_time / iterations;
    
    const elements_per_iter = shape.numel();
    const total_elements = elements_per_iter * iterations;
    const ops_per_second = @as(f64, @floatFromInt(total_elements)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0);
    
    return BenchmarkResult{
        .name = "Tensor Addition (SIMD)",
        .iterations = iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .ops_per_second = ops_per_second,
        .memory_used_mb = (4096.0 * 1024.0 * 4.0 * 3.0) / (1024.0 * 1024.0), // 3 tensors
    };
}

/// Benchmark matrix multiplication performance
fn benchmarkMatrixMultiplication(allocator: std.mem.Allocator) !BenchmarkResult {
    const iterations = 10;
    const m = 1024;
    const k = 1024;
    const n = 1024;
    
    const a_shape = Shape.init(&[_]u32{ m, k });
    const b_shape = Shape.init(&[_]u32{ k, n });
    const c_shape = Shape.init(&[_]u32{ m, n });
    
    var a = try deepseek_core.Tensor.ones(allocator, a_shape, .f32);
    defer a.deinit();
    
    var b = try deepseek_core.Tensor.ones(allocator, b_shape, .f32);
    defer b.deinit();
    
    var c = try deepseek_core.Tensor.zeros(allocator, c_shape, .f32);
    defer c.deinit();
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        try a.matmul(&b, &c);
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    const avg_time = total_time / iterations;
    
    // FLOPS calculation: 2 * M * N * K operations per matrix multiplication
    const flops_per_iter = 2 * m * n * k;
    const total_flops = flops_per_iter * iterations;
    const gflops_per_second = (@as(f64, @floatFromInt(total_flops)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0)) / 1_000_000_000.0;
    
    return BenchmarkResult{
        .name = "Matrix Multiplication",
        .iterations = iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .ops_per_second = gflops_per_second, // Actually GFLOPS
        .memory_used_mb = (@as(f64, @floatFromInt(m + k + n)) * 1024.0 * 4.0) / (1024.0 * 1024.0),
    };
}

/// Benchmark SwiGLU activation function
fn benchmarkSwiGLU(allocator: std.mem.Allocator) !BenchmarkResult {
    const iterations = 1000;
    const size = 1024 * 1024; // 1M elements
    
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    
    const gate = try allocator.alloc(f32, size);
    defer allocator.free(gate);
    
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);
    
    // Fill with random data
    for (input, gate) |*i, *g| {
        i.* = 0.5;
        g.* = 0.3;
    }
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        // SwiGLU: input * swish(gate)
        for (0..size) |i| {
            const g = gate[i];
            const swish_g = g / (1.0 + @exp(-g));
            output[i] = input[i] * swish_g;
        }
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    const avg_time = total_time / iterations;
    
    const total_elements = size * iterations;
    const ops_per_second = @as(f64, @floatFromInt(total_elements)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0);
    
    return BenchmarkResult{
        .name = "SwiGLU Activation",
        .iterations = iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .ops_per_second = ops_per_second,
        .memory_used_mb = (@as(f64, @floatFromInt(size)) * 3.0 * 4.0) / (1024.0 * 1024.0),
    };
}

/// Benchmark RMS normalization
fn benchmarkRMSNorm(allocator: std.mem.Allocator) !BenchmarkResult {
    const iterations = 1000;
    const size = 4096; // Typical hidden dimension
    
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    
    const weight = try allocator.alloc(f32, size);
    defer allocator.free(weight);
    
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);
    
    // Initialize data
    for (input, weight) |*i, *w| {
        i.* = 0.1;
        w.* = 1.0;
    }
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        deepseek_core.math.rms_norm.rmsNormVec(input, weight, output, 1e-6);
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    const avg_time = total_time / iterations;
    
    const ops_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0);
    
    return BenchmarkResult{
        .name = "RMS Normalization (SIMD)",
        .iterations = iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .ops_per_second = ops_per_second,
        .memory_used_mb = (@as(f64, @floatFromInt(size)) * 3.0 * 4.0) / (1024.0 * 1024.0),
    };
}

/// Benchmark memory bandwidth
fn benchmarkMemoryBandwidth(allocator: std.mem.Allocator) !BenchmarkResult {
    const iterations = 100;
    const size = 64 * 1024 * 1024; // 64MB
    
    const source = try allocator.alloc(u8, size);
    defer allocator.free(source);
    
    const dest = try allocator.alloc(u8, size);
    defer allocator.free(dest);
    
    // Fill source with data
    @memset(source, 0x42);
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        @memcpy(dest, source);
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    const avg_time = total_time / iterations;
    
    const total_bytes = size * iterations;
    const gb_per_second = (@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    return BenchmarkResult{
        .name = "Memory Bandwidth",
        .iterations = iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .ops_per_second = gb_per_second, // Actually GB/s
        .memory_used_mb = (@as(f64, @floatFromInt(size)) * 2.0) / (1024.0 * 1024.0),
    };
} 