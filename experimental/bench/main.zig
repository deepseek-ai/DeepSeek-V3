// Benchmark Suite for DeepZig V3 Implementation
// Tests performance of core operations across different backends

const std = @import("std");
const print = std.debug.print;

const cpu_backend = @import("cpu_backend");
const deepseek_core = @import("deepseek_core");
const Shape = deepseek_core.Shape;

// Benchmark result collection
const MatrixResult = struct {
    size: u32,
    gflops: f64,
    time_ms: f64,
};

const BenchmarkResults = struct {
    matrix_results: std.ArrayList(MatrixResult),
    tensor_add_bandwidth_gbps: f64,
    memory_copy_bandwidth_gbps: f64,
    memory_latency_ns: f64,
    blas_backend: ?[]const u8,
    blas_peak_gflops: f64,

    pub fn init(allocator: std.mem.Allocator) BenchmarkResults {
        return BenchmarkResults{
            .matrix_results = std.ArrayList(MatrixResult).init(allocator),
            .tensor_add_bandwidth_gbps = 0,
            .memory_copy_bandwidth_gbps = 0,
            .memory_latency_ns = 0,
            .blas_backend = null,
            .blas_peak_gflops = 0,
        };
    }

    pub fn deinit(self: *BenchmarkResults) void {
        self.matrix_results.deinit();
    }

    pub fn setBLASBackend(self: *BenchmarkResults, backend: anytype) void {
        switch (backend) {
            .naive => self.blas_backend = "Naive",
            .accelerate => self.blas_backend = "Apple Accelerate",
            .intel_mkl => self.blas_backend = "Intel MKL",
            .openblas => self.blas_backend = "OpenBLAS",
        }
    }
};

// Import Shape from deepseek_core
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
        try writer.print("{s:30} | {d:6} iter | {d:8.2} ms | {d:10.0} ops/s | {d:6.1} MB", .{ self.name, self.iterations, @as(f64, @floatFromInt(self.avg_time_ns)) / 1_000_000.0, self.ops_per_second, self.memory_used_mb });
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize results collection
    var results = BenchmarkResults.init(allocator);
    defer results.deinit();

    // Print banner
    printBanner();

    // Run comprehensive benchmarks and collect results
    try runTensorBenchmarks(allocator, &results);
    try runBlasBenchmarks(allocator, &results);
    try runMemoryBenchmarks(allocator, &results);

    // Print dynamic summary based on actual results
    printDynamicSummary(&results);

    std.log.info("🎉 Benchmark suite completed!", .{});
}

fn printBanner() void {
    std.log.info("🚀 DeepZig V3 Performance Benchmarks", .{});
    std.log.info("==========================================", .{});
    std.log.info("", .{});
}

fn runTensorBenchmarks(allocator: std.mem.Allocator, results: *BenchmarkResults) !void {
    std.log.info("📊 TENSOR OPERATIONS BENCHMARK", .{});
    std.log.info("-------------------------------", .{});

    // Test different matrix sizes
    const sizes = [_]u32{ 256, 512, 1024, 2048 };
    const iterations = [_]u32{ 50, 20, 10, 5 };

    for (sizes, iterations) |size, iters| {
        try benchmarkMatrixMultiplication(allocator, size, iters, results);
    }

    // Tensor addition benchmark
    try benchmarkTensorAddition(allocator, results);

    std.log.info("", .{});
}

fn benchmarkMatrixMultiplication(allocator: std.mem.Allocator, size: u32, iterations: u32, results: *BenchmarkResults) !void {
    std.log.info("🔢 Matrix Multiplication {}x{} ({} iterations)", .{ size, size, iterations });

    // Create matrices
    var a = try deepseek_core.createMatrix(.f32, allocator, size, size);
    var b = try deepseek_core.createMatrix(.f32, allocator, size, size);
    var c = try deepseek_core.createMatrix(.f32, allocator, size, size);
    defer a.deinit();
    defer b.deinit();
    defer c.deinit();

    // Fill with random data
    a.fillRandom(42);
    b.fillRandom(123);

    // Benchmark
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try a.matmul(&b, &c);
    }
    const elapsed_ns = timer.read();

    // Calculate performance metrics
    const ops = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(iterations));
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const gflops = ops / elapsed_s / 1e9;
    const avg_time_ms = elapsed_s * 1000.0 / @as(f64, @floatFromInt(iterations));

    // Performance comparison
    if (a.blas_ctx) |blas_context| {
        const efficiency = gflops / blas_context.performance_info.peak_gflops * 100.0;
        std.log.info("  ✅ BLAS-accelerated: {d:.1} ms/iter, {d:.1} GFLOPS ({d:.1}% efficiency)", .{ avg_time_ms, gflops, efficiency });
        std.log.info("  🔧 Backend: {}, Peak: {d:.1} GFLOPS", .{ blas_context.backend, blas_context.performance_info.peak_gflops });
        try results.matrix_results.append(MatrixResult{
            .size = size,
            .gflops = gflops,
            .time_ms = avg_time_ms,
        });
    } else {
        std.log.info("  ⚠️ Naive implementation: {d:.1} ms/iter, {d:.1} GFLOPS", .{ avg_time_ms, gflops });
    }
}

fn benchmarkTensorAddition(allocator: std.mem.Allocator, results: *BenchmarkResults) !void {
    const size = 1024 * 1024; // 1M elements
    const iterations = 1000;

    std.log.info("➕ Tensor Addition (SIMD) - {} elements, {} iterations", .{ size, iterations });

    var a = try deepseek_core.createVector(.f32, allocator, size);
    var b = try deepseek_core.createVector(.f32, allocator, size);
    var c = try deepseek_core.createVector(.f32, allocator, size);
    defer a.deinit();
    defer b.deinit();
    defer c.deinit();

    a.fillRandom(42);
    b.fillRandom(123);

    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try a.add(&b, &c);
    }
    const elapsed_ns = timer.read();

    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const operations_per_sec = @as(f64, @floatFromInt(size * iterations)) / elapsed_s;
    const bandwidth_gb_s = operations_per_sec * @sizeOf(f32) * 3 / (1024 * 1024 * 1024); // 3x for read a, read b, write c

    std.log.info("  ✅ {d:.1} GOp/s, {d:.1} GB/s bandwidth", .{ operations_per_sec / 1e9, bandwidth_gb_s });
    results.tensor_add_bandwidth_gbps = bandwidth_gb_s;
}

fn runBlasBenchmarks(allocator: std.mem.Allocator, results: *BenchmarkResults) !void {
    std.log.info("🧮 BLAS LIBRARY BENCHMARK", .{});
    std.log.info("-------------------------", .{});

    // Initialize BLAS and show detection results
    const blas_context = deepseek_core.blas.Blas.init(allocator) catch {
        std.log.info("⚠️ BLAS initialization failed, using naive implementation", .{});
        return;
    };

    std.log.info("🔍 BLAS Detection Results:", .{});
    std.log.info("  Backend: {}", .{blas_context.backend});
    std.log.info("  Expected Peak Performance: {d:.1} GFLOPS", .{blas_context.performance_info.peak_gflops});
    std.log.info("  Memory Bandwidth: {d:.1} GB/s", .{blas_context.performance_info.memory_bandwidth_gb_s});
    std.log.info("  SIMD Width: {} bits", .{blas_context.performance_info.simd_width});
    std.log.info("  Mixed Precision: {}", .{blas_context.performance_info.supports_mixed_precision});

    // Run dedicated BLAS benchmark
    std.log.info("", .{});
    std.log.info("🚀 Running dedicated BLAS benchmark...", .{});
    try deepseek_core.blas.benchmarkBlas(allocator);

    std.log.info("", .{});
    results.setBLASBackend(blas_context.backend);
    results.blas_peak_gflops = blas_context.performance_info.peak_gflops;
}

fn runMemoryBenchmarks(allocator: std.mem.Allocator, results: *BenchmarkResults) !void {
    std.log.info("💾 MEMORY PERFORMANCE BENCHMARK", .{});
    std.log.info("--------------------------------", .{});

    try benchmarkMemoryBandwidth(allocator, results);
    try benchmarkMemoryLatency(allocator, results);

    std.log.info("", .{});
}

fn benchmarkMemoryBandwidth(allocator: std.mem.Allocator, results: *BenchmarkResults) !void {
    const size = 128 * 1024 * 1024 / @sizeOf(f32); // 128MB of f32s
    const iterations = 100;

    std.log.info("📈 Memory Bandwidth Test - {} MB, {} iterations", .{ size * @sizeOf(f32) / (1024 * 1024), iterations });

    const data = try allocator.alloc(f32, size);
    defer allocator.free(data);

    // Fill with data
    for (data, 0..) |*ptr, i| {
        ptr.* = @floatFromInt(i % 1000);
    }

    // Sequential read benchmark
    var timer = try std.time.Timer.start();
    var checksum: f64 = 0;
    for (0..iterations) |_| {
        for (data) |value| {
            checksum += value;
        }
    }
    const elapsed_ns = timer.read();

    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const bytes_read = @as(f64, @floatFromInt(size * @sizeOf(f32) * iterations));
    const bandwidth_gb_s = bytes_read / elapsed_s / (1024 * 1024 * 1024);

    std.log.info("  ✅ Sequential Read: {d:.1} GB/s (checksum: {d:.1})", .{ bandwidth_gb_s, checksum });

    // Memory copy benchmark
    const dest = try allocator.alloc(f32, size);
    defer allocator.free(dest);

    timer.reset();
    for (0..iterations) |_| {
        @memcpy(dest, data);
    }
    const copy_elapsed_ns = timer.read();

    const copy_elapsed_s = @as(f64, @floatFromInt(copy_elapsed_ns)) / 1e9;
    const copy_bandwidth_gb_s = bytes_read / copy_elapsed_s / (1024 * 1024 * 1024);

    std.log.info("  ✅ Memory Copy: {d:.1} GB/s", .{copy_bandwidth_gb_s});
    results.memory_copy_bandwidth_gbps = copy_bandwidth_gb_s;
}

fn benchmarkMemoryLatency(allocator: std.mem.Allocator, results: *BenchmarkResults) !void {
    const size = 1024 * 1024; // 1M elements
    const iterations = 1000;

    std.log.info("⏱️ Memory Latency Test - Random Access Pattern", .{});

    const data = try allocator.alloc(u32, size);
    defer allocator.free(data);

    // Create random access pattern
    var rng = std.Random.DefaultPrng.init(42);
    for (data, 0..) |*ptr, i| {
        ptr.* = @intCast(rng.random().uintLessThan(usize, size));
        _ = i;
    }

    var timer = try std.time.Timer.start();
    var index: u32 = 0;
    for (0..iterations) |_| {
        for (0..size) |_| {
            index = data[index];
        }
    }
    const elapsed_ns = timer.read();

    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const accesses_per_sec = @as(f64, @floatFromInt(size * iterations)) / elapsed_s;
    const avg_latency_ns = elapsed_s * 1e9 / @as(f64, @floatFromInt(size * iterations));

    std.log.info("  ✅ {d:.1} M accesses/s, {d:.1} ns avg latency (index: {})", .{ accesses_per_sec / 1e6, avg_latency_ns, index });
    results.memory_latency_ns = avg_latency_ns;
}

fn printDynamicSummary(results: *BenchmarkResults) void {
    std.log.info("", .{});
    std.log.info("🎯 DYNAMIC BENCHMARK SUMMARY", .{});
    std.log.info("===============================", .{});
    std.log.info("", .{});

    if (results.matrix_results.items.len > 0) {
        std.log.info("📊 Matrix Multiplication Performance:", .{});
        for (results.matrix_results.items) |result| {
            std.log.info("  • {}×{}: {d:.1} ms, {d:.0} GFLOPS", .{ result.size, result.size, result.time_ms, result.gflops });
        }

        // Find best performance
        var best_gflops: f64 = 0;
        var best_size: u32 = 0;
        for (results.matrix_results.items) |result| {
            if (result.gflops > best_gflops) {
                best_gflops = result.gflops;
                best_size = result.size;
            }
        }
        std.log.info("  🏆 Peak measured: {d:.0} GFLOPS at {}×{}", .{ best_gflops, best_size, best_size });
        std.log.info("", .{});
    }

    if (results.blas_backend) |backend_name| {
        std.log.info("🧮 BLAS Configuration:", .{});
        std.log.info("  • Backend: {s}", .{backend_name});
        std.log.info("  • Theoretical peak: {d:.0} GFLOPS (estimated)", .{results.blas_peak_gflops});
        std.log.info("", .{});
    }

    if (results.tensor_add_bandwidth_gbps > 0) {
        std.log.info("➕ Tensor Operations:", .{});
        std.log.info("  • SIMD Addition: {d:.1} GB/s", .{results.tensor_add_bandwidth_gbps});
        std.log.info("", .{});
    }

    if (results.memory_copy_bandwidth_gbps > 0 or results.memory_latency_ns > 0) {
        std.log.info("💾 Memory Performance:", .{});
        if (results.memory_copy_bandwidth_gbps > 0) {
            std.log.info("  • Copy Bandwidth: {d:.1} GB/s", .{results.memory_copy_bandwidth_gbps});
        }
        if (results.memory_latency_ns > 0) {
            std.log.info("  • Random Access Latency: {d:.1} ns", .{results.memory_latency_ns});
        }
        std.log.info("", .{});
    }

    // Performance assessment based on actual measurements only
    if (results.matrix_results.items.len > 0) {
        var best_measured_gflops: f64 = 0;
        for (results.matrix_results.items) |result| {
            if (result.gflops > best_measured_gflops) {
                best_measured_gflops = result.gflops;
            }
        }

        std.log.info("🎯 Performance Assessment:", .{});

        if (best_measured_gflops > 1000) {
            std.log.info("  ✅ Excellent: BLAS delivering 1000+ GFLOPS", .{});
        } else if (best_measured_gflops > 500) {
            std.log.info("  ✅ Good: BLAS delivering 500+ GFLOPS", .{});
        } else if (best_measured_gflops > 100) {
            std.log.info("  ⚠️ Moderate: BLAS working, performance could improve", .{});
        } else {
            std.log.info("  ❌ Poor: BLAS may not be working optimally", .{});
        }

        // Only show efficiency comparison if we have reasonable confidence in the estimate
        if (results.blas_peak_gflops > best_measured_gflops * 1.5) {
            const estimated_efficiency = best_measured_gflops / results.blas_peak_gflops * 100.0;
            std.log.info("  • Est. efficiency: {d:.0}% (vs theoretical peak)", .{estimated_efficiency});
        }

        std.log.info("", .{});
    }
}
