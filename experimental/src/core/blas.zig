// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

// High-Performance BLAS Integration for DeepZig V3
// Automatically detects and uses the fastest BLAS implementation per platform
//
// Performance targets:
// - Apple Silicon (M1/M2/M3/M4): Accelerate.framework (~2000 GFLOPS)
// - Intel/AMD x86_64: Intel MKL or OpenBLAS (~1000+ GFLOPS)
// - ARM64 Linux: OpenBLAS with NEON (~500+ GFLOPS)
// - Fallback: Naive implementation (~10 GFLOPS)

const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.Random;
const builtin = @import("builtin");

/// Simple Apple Silicon detection for BLAS optimization
fn isAppleSilicon() bool {
    return builtin.os.tag == .macos and builtin.target.cpu.arch == .aarch64;
}

/// BLAS backend selection based on platform and hardware capabilities
pub const BlasBackend = enum {
    accelerate, // macOS Accelerate.framework (Apple Silicon & Intel)
    intel_mkl, // Intel Math Kernel Library (x86_64)
    openblas, // OpenBLAS (cross-platform, good ARM64 support)
    naive, // Fallback pure Zig implementation

    /// Automatically detect the optimal BLAS backend for current platform
    pub fn detectOptimal(allocator: Allocator) BlasBackend {
        _ = allocator; // Mark unused parameter
        return switch (builtin.os.tag) {
            .macos => .accelerate, // Always use Accelerate on macOS
            .linux => detectLinuxOptimal(),
            .windows => detectWindowsOptimal(),
            else => .naive,
        };
    }

    fn detectLinuxOptimal() BlasBackend {
        // Prefer Intel MKL on Intel CPUs, OpenBLAS elsewhere
        if (builtin.cpu.arch == .x86_64) {
            // Check if Intel MKL is available (could add runtime detection)
            return .openblas; // Default to OpenBLAS for broader compatibility
        } else {
            return .openblas; // OpenBLAS has excellent ARM64/NEON support
        }
    }

    fn detectWindowsOptimal() BlasBackend {
        return switch (builtin.cpu.arch) {
            .x86_64 => .openblas, // OpenBLAS is most portable on Windows
            else => .naive,
        };
    }

    /// Get expected performance characteristics for this backend
    pub fn getPerformanceInfo(self: BlasBackend, allocator: Allocator) BlasPerformanceInfo {
        _ = allocator; // Mark unused parameter
        return switch (self) {
            .accelerate => blk: {
                // Basic Apple Silicon detection for performance estimation
                const gflops: f32 = if (isAppleSilicon()) 2600 else 1000; // Estimate M1-level performance

                break :blk .{
                    .peak_gflops = gflops,
                    .memory_bandwidth_gb_s = 200,
                    .supports_mixed_precision = true,
                    .simd_width = 128, // NEON 128-bit
                };
            },
            .intel_mkl => .{
                .peak_gflops = 1500,
                .memory_bandwidth_gb_s = 100,
                .supports_mixed_precision = true,
                .simd_width = 512, // AVX-512
            },
            .openblas => .{
                .peak_gflops = 800,
                .memory_bandwidth_gb_s = 80,
                .supports_mixed_precision = false,
                .simd_width = if (builtin.cpu.arch == .aarch64) 128 else 256,
            },
            .naive => .{
                .peak_gflops = 10,
                .memory_bandwidth_gb_s = 20,
                .supports_mixed_precision = false,
                .simd_width = 128,
            },
        };
    }
};

pub const BlasPerformanceInfo = struct {
    peak_gflops: f32,
    memory_bandwidth_gb_s: f32,
    supports_mixed_precision: bool,
    simd_width: u32,
};

/// Matrix dimensions for BLAS operations
pub const MatrixDims = struct {
    m: u32, // rows of A and C
    n: u32, // cols of B and C
    k: u32, // cols of A, rows of B
};

/// Memory layout for matrices
pub const MatrixLayout = enum {
    row_major, // C-style (row by row)
    column_major, // Fortran-style (column by column)
};

/// Transpose operations
pub const Transpose = enum {
    no_trans,
    trans,
    conj_trans, // For complex numbers

    fn toCblas(self: Transpose) c_int {
        return switch (self) {
            .no_trans => 111, // CblasNoTrans
            .trans => 112, // CblasTrans
            .conj_trans => 113, // CblasConjTrans
        };
    }
};

// Platform-specific FFI declarations
const blas_c = switch (builtin.os.tag) {
    .macos => struct {
        // macOS Accelerate.framework
        extern "c" fn cblas_sgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: [*]const f32,
            lda: c_int,
            b: [*]const f32,
            ldb: c_int,
            beta: f32,
            result: [*]f32,
            ldc: c_int,
        ) void;

        extern "c" fn cblas_dgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f64,
            a: [*]const f64,
            lda: c_int,
            b: [*]const f64,
            ldb: c_int,
            beta: f64,
            result: [*]f64,
            ldc: c_int,
        ) void;
    },
    else => struct {
        // OpenBLAS or Intel MKL (same CBLAS interface)
        extern "c" fn cblas_sgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: [*]const f32,
            lda: c_int,
            b: [*]const f32,
            ldb: c_int,
            beta: f32,
            result: [*]f32,
            ldc: c_int,
        ) void;

        extern "c" fn cblas_dgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f64,
            a: [*]const f64,
            lda: c_int,
            b: [*]const f64,
            ldb: c_int,
            beta: f64,
            result: [*]f64,
            ldc: c_int,
        ) void;
    },
};

/// High-level BLAS interface - automatically chooses optimal implementation
pub const Blas = struct {
    backend: BlasBackend,
    performance_info: BlasPerformanceInfo,
    allocator: Allocator,

    /// Initialize BLAS with optimal backend detection
    pub fn init(allocator: Allocator) !Blas {
        const backend = BlasBackend.detectOptimal(allocator);
        const performance_info = backend.getPerformanceInfo(allocator);

        std.log.info("BLAS initialized with {} backend", .{backend});
        std.log.info("Expected performance: {d:.1} GFLOPS, {d:.1} GB/s bandwidth", .{
            performance_info.peak_gflops,
            performance_info.memory_bandwidth_gb_s,
        });

        return Blas{
            .backend = backend,
            .performance_info = performance_info,
            .allocator = allocator,
        };
    }

    /// Single-precision matrix multiplication: C = alpha * A * B + beta * C
    pub fn sgemm(
        self: *const Blas,
        layout: MatrixLayout,
        transa: Transpose,
        transb: Transpose,
        dims: MatrixDims,
        alpha: f32,
        a: []const f32,
        b: []const f32,
        beta: f32,
        result: []f32,
    ) void {
        switch (self.backend) {
            .accelerate, .intel_mkl, .openblas => {
                const order: c_int = if (layout == .row_major) 101 else 102; // CblasRowMajor : CblasColMajor
                const lda = if (layout == .row_major) @as(c_int, @intCast(dims.k)) else @as(c_int, @intCast(dims.m));
                const ldb = if (layout == .row_major) @as(c_int, @intCast(dims.n)) else @as(c_int, @intCast(dims.k));
                const ldc = if (layout == .row_major) @as(c_int, @intCast(dims.n)) else @as(c_int, @intCast(dims.m));

                blas_c.cblas_sgemm(
                    order,
                    transa.toCblas(),
                    transb.toCblas(),
                    @intCast(dims.m),
                    @intCast(dims.n),
                    @intCast(dims.k),
                    alpha,
                    a.ptr,
                    lda,
                    b.ptr,
                    ldb,
                    beta,
                    result.ptr,
                    ldc,
                );
            },
            .naive => {
                naiveSgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
            },
        }
    }

    /// Double-precision matrix multiplication: C = alpha * A * B + beta * C
    pub fn dgemm(
        self: *const Blas,
        layout: MatrixLayout,
        transa: Transpose,
        transb: Transpose,
        dims: MatrixDims,
        alpha: f64,
        a: []const f64,
        b: []const f64,
        beta: f64,
        result: []f64,
    ) void {
        switch (self.backend) {
            .accelerate, .intel_mkl, .openblas => {
                const order: c_int = if (layout == .row_major) 101 else 102;
                const lda = if (layout == .row_major) @as(c_int, @intCast(dims.k)) else @as(c_int, @intCast(dims.m));
                const ldb = if (layout == .row_major) @as(c_int, @intCast(dims.n)) else @as(c_int, @intCast(dims.k));
                const ldc = if (layout == .row_major) @as(c_int, @intCast(dims.n)) else @as(c_int, @intCast(dims.m));

                blas_c.cblas_dgemm(
                    order,
                    transa.toCblas(),
                    transb.toCblas(),
                    @intCast(dims.m),
                    @intCast(dims.n),
                    @intCast(dims.k),
                    alpha,
                    a.ptr,
                    lda,
                    b.ptr,
                    ldb,
                    beta,
                    result.ptr,
                    ldc,
                );
            },
            .naive => {
                naiveDgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
            },
        }
    }

    /// Generic matrix multiplication (chooses sgemm or dgemm based on type)
    pub fn matmul(self: *const Blas, comptime T: type, a: []const T, b: []const T, result: []T, dims: MatrixDims) void {
        switch (T) {
            f32 => self.sgemm(.row_major, .no_trans, .no_trans, dims, 1.0, a, b, 0.0, result),
            f64 => self.dgemm(.row_major, .no_trans, .no_trans, dims, 1.0, a, b, 0.0, result),
            else => @compileError("BLAS matmul only supports f32 and f64"),
        }
    }
};

// Naive BLAS implementations for fallback
fn naiveSgemm(
    layout: MatrixLayout,
    transa: Transpose,
    transb: Transpose,
    dims: MatrixDims,
    alpha: f32,
    a: []const f32,
    b: []const f32,
    beta: f32,
    result: []f32,
) void {
    _ = layout;
    _ = transa;
    _ = transb; // TODO: Handle these properly

    // Simple case: C = alpha * A * B + beta * C (no transpose)
    const m = dims.m;
    const n = dims.n;
    const k = dims.k;

    // Scale existing C by beta
    for (result) |*val| {
        val.* *= beta;
    }

    // Add alpha * A * B
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |l| {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] += alpha * sum;
        }
    }
}

fn naiveDgemm(
    layout: MatrixLayout,
    transa: Transpose,
    transb: Transpose,
    dims: MatrixDims,
    alpha: f64,
    a: []const f64,
    b: []const f64,
    beta: f64,
    result: []f64,
) void {
    _ = layout;
    _ = transa;
    _ = transb; // TODO: Handle these properly

    const m = dims.m;
    const n = dims.n;
    const k = dims.k;

    // Scale existing C by beta
    for (result) |*val| {
        val.* *= beta;
    }

    // Add alpha * A * B
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f64 = 0.0;
            for (0..k) |l| {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] += alpha * sum;
        }
    }
}

/// Helper function to create matrix and fill with test data
pub fn createMatrix(comptime T: type, allocator: Allocator, rows: usize, cols: usize) ![]T {
    return try allocator.alloc(T, rows * cols);
}

/// Benchmark BLAS performance
pub fn benchmarkBlas(allocator: Allocator) !void {
    const size = 1024;
    const iterations = 10;

    std.log.info("🚀 Benchmarking BLAS operations ({}x{} matrices, {} iterations)...", .{ size, size, iterations });

    // Initialize BLAS
    const blas = try Blas.init(allocator);

    // Create test matrices
    const matrix_a = try createMatrix(f32, allocator, size, size);
    const matrix_b = try createMatrix(f32, allocator, size, size);
    const matrix_c = try createMatrix(f32, allocator, size, size);
    defer allocator.free(matrix_a);
    defer allocator.free(matrix_b);
    defer allocator.free(matrix_c);

    // Fill with random data
    var prng = Random.DefaultPrng.init(42);
    const random = prng.random();
    for (matrix_a) |*val| val.* = random.float(f32);
    for (matrix_b) |*val| val.* = random.float(f32);
    @memset(matrix_c, 0.0);

    // Benchmark matrix multiplication
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });
    }
    const elapsed_ns = timer.read();

    const ops = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(iterations));
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const gflops = ops / elapsed_s / 1e9;

    std.log.info("✅ BLAS Matrix Multiplication Results:", .{});
    std.log.info("  Time: {d:.3} ms", .{elapsed_s * 1000.0});
    std.log.info("  Performance: {d:.1} GFLOPS", .{gflops});
    std.log.info("  Backend: {}", .{blas.backend});

    const efficiency = gflops / blas.performance_info.peak_gflops * 100.0;
    std.log.info("  Efficiency: {d:.1}% of peak BLAS performance", .{efficiency});
}

// Basic tests
test "BLAS initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const blas = try Blas.init(allocator);
    try std.testing.expect(blas.performance_info.peak_gflops > 0);
}

test "matrix multiplication correctness" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const blas = try Blas.init(allocator);

    // Test 2x2 matrix multiplication
    var matrix_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var matrix_b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var matrix_c = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    blas.matmul(f32, &matrix_a, &matrix_b, &matrix_c, .{ .m = 2, .n = 2, .k = 2 });

    // Expected result: C = [[19, 22], [43, 50]]
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), matrix_c[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), matrix_c[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), matrix_c[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), matrix_c[3], 1e-6);
}
