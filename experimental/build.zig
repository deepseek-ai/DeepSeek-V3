// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable
    const exe = b.addExecutable(.{
        .name = "deepseek-v3-zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // BLAS library configuration based on target platform
    configureBlas(exe, target);

    // Add module dependencies
    const deepseek_core = b.addModule("deepseek_core", .{
        .root_source_file = b.path("src/core/root.zig"),
    });
    exe.root_module.addImport("deepseek_core", deepseek_core);

    const web_layer = b.addModule("web_layer", .{
        .root_source_file = b.path("src/web/root.zig"),
    });
    web_layer.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("web_layer", web_layer);

    const cpu_backend = b.addModule("cpu_backend", .{
        .root_source_file = b.path("src/backends/cpu/root.zig"),
    });
    cpu_backend.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("cpu_backend", cpu_backend);

    const metal_backend = b.addModule("metal_backend", .{
        .root_source_file = b.path("src/backends/metal/root.zig"),
    });
    metal_backend.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("metal_backend", metal_backend);

    // Add Metal framework for macOS
    if (target.result.os.tag == .macos) {
        exe.linkFramework("Metal");
        exe.linkFramework("Foundation");
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Benchmarks
    const benchmark_exe = b.addExecutable(.{
        .name = "deepseek-v3-benchmark",
        .root_source_file = b.path("bench/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add the same modules to benchmark
    benchmark_exe.root_module.addImport("deepseek_core", deepseek_core);

    const cpu_backend_bench = b.addModule("cpu_backend", .{
        .root_source_file = b.path("src/backends/cpu/root.zig"),
    });
    cpu_backend_bench.addImport("deepseek_core", deepseek_core);
    benchmark_exe.root_module.addImport("cpu_backend", cpu_backend_bench);

    // Configure BLAS for benchmarks too
    configureBlas(benchmark_exe, target);

    // Add Metal framework for benchmarks on macOS
    if (target.result.os.tag == .macos) {
        benchmark_exe.linkFramework("Metal");
        benchmark_exe.linkFramework("Foundation");
    }

    b.installArtifact(benchmark_exe);

    const benchmark_run_cmd = b.addRunArtifact(benchmark_exe);
    benchmark_run_cmd.step.dependOn(b.getInstallStep());

    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&benchmark_run_cmd.step);

    // BLAS benchmarks specifically
    const blas_bench_exe = b.addExecutable(.{
        .name = "blas-benchmark",
        .root_source_file = b.path("bench/blas_bench.zig"),
        .target = target,
        .optimize = optimize,
    });

    blas_bench_exe.root_module.addImport("deepseek_core", deepseek_core);
    configureBlas(blas_bench_exe, target);

    const blas_bench_run = b.addRunArtifact(blas_bench_exe);
    const blas_bench_step = b.step("bench-blas", "Run BLAS-specific benchmarks");
    blas_bench_step.dependOn(&blas_bench_run.step);
}

/// Configure BLAS linking for the given compile step based on target platform
fn configureBlas(step: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    const target_os = target.result.os.tag;

    switch (target_os) {
        .macos => {
            // Use Apple's Accelerate framework
            step.linkFramework("Accelerate");
            step.root_module.addCMacro("HAVE_ACCELERATE", "1");
        },
        .linux => {
            // Use OpenBLAS on Linux
            step.linkSystemLibrary("openblas");
            step.root_module.addCMacro("HAVE_OPENBLAS", "1");
        },
        .windows => {
            // Use OpenBLAS on Windows (if available)
            step.linkSystemLibrary("openblas");
            step.root_module.addCMacro("HAVE_OPENBLAS", "1");
        },
        else => {
            // Fallback to naive implementation
            step.root_module.addCMacro("HAVE_NAIVE_BLAS", "1");
        },
    }
}
