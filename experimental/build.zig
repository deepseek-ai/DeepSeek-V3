const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard optimization options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // === CORE LIBRARY MODULE ===
    const deepseek_core = b.addModule("deepseek_core", .{
        .root_source_file = b.path("src/core/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === WEB LAYER MODULE ===
    const web_layer = b.addModule("web_layer", .{
        .root_source_file = b.path("src/web/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    web_layer.addImport("deepseek_core", deepseek_core);

    // === BACKEND MODULES ===
    const cpu_backend = b.addModule("cpu_backend", .{
        .root_source_file = b.path("src/backends/cpu/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    cpu_backend.addImport("deepseek_core", deepseek_core);

    const metal_backend = b.addModule("metal_backend", .{
        .root_source_file = b.path("src/backends/metal/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    metal_backend.addImport("deepseek_core", deepseek_core);

    const cuda_backend = b.addModule("cuda_backend", .{
        .root_source_file = b.path("src/backends/cuda/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    cuda_backend.addImport("deepseek_core", deepseek_core);

    // === MAIN EXECUTABLE ===
    const exe = b.addExecutable(.{
        .name = "deepseek-v3-zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add imports to main executable
    exe.root_module.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("web_layer", web_layer);
    exe.root_module.addImport("cpu_backend", cpu_backend);
    exe.root_module.addImport("metal_backend", metal_backend);
    exe.root_module.addImport("cuda_backend", cuda_backend);

    // Platform-specific backend linking
    if (target.result.os.tag == .macos) {
        exe.linkFramework("Metal");
        exe.linkFramework("MetalKit");
        exe.linkFramework("Foundation");
    }

    // CUDA linking for Linux/Windows
    if (target.result.os.tag == .linux or target.result.os.tag == .windows) {
        // TODO: Add CUDA library paths when available
        // exe.addLibraryPath(b.path("cuda/lib"));
        // exe.linkSystemLibrary("cuda");
        // exe.linkSystemLibrary("cublas");
    }

    b.installArtifact(exe);

    // === RUN COMMAND ===
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the DeepSeek V3 server");
    run_step.dependOn(&run_cmd.step);

    // === TESTING ===
    const test_step = b.step("test", "Run unit tests");

    // Core tests
    const core_tests = b.addTest(.{
        .root_source_file = b.path("src/core/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_step.dependOn(&b.addRunArtifact(core_tests).step);

    // Web tests
    const web_tests = b.addTest(.{
        .root_source_file = b.path("src/web/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    web_tests.root_module.addImport("deepseek_core", deepseek_core);
    test_step.dependOn(&b.addRunArtifact(web_tests).step);

    // Backend tests
    const cpu_tests = b.addTest(.{
        .root_source_file = b.path("src/backends/cpu/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    cpu_tests.root_module.addImport("deepseek_core", deepseek_core);
    test_step.dependOn(&b.addRunArtifact(cpu_tests).step);

    // === BENCHMARKS ===
    const bench_step = b.step("bench", "Run benchmarks");
    
    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_source_file = b.path("bench/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_exe.root_module.addImport("deepseek_core", deepseek_core);
    bench_exe.root_module.addImport("cpu_backend", cpu_backend);
    
    const bench_run = b.addRunArtifact(bench_exe);
    bench_step.dependOn(&bench_run.step);

    // === WASM TARGET ===
    const wasm_step = b.step("wasm", "Build WebAssembly target");
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });
    
    const wasm_exe = b.addExecutable(.{
        .name = "deepseek-v3-wasm",
        .root_source_file = b.path("src/wasm/main.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_exe.root_module.addImport("deepseek_core", deepseek_core);
    wasm_exe.entry = .disabled;
    wasm_exe.rdynamic = true;
    
    const wasm_install = b.addInstallArtifact(wasm_exe, .{});
    wasm_step.dependOn(&wasm_install.step);
} 