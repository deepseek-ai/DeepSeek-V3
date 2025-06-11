// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;

const cpu_backend = @import("cpu_backend");
const deepseek_core = @import("deepseek_core");
const metal_backend = @import("metal_backend");
const web_layer = @import("web_layer");

const Config = struct {
    port: u16 = 8080,
    host: []const u8 = "127.0.0.1",
    model_path: ?[]const u8 = null,
    backend: Backend = .cpu,
    max_concurrent_requests: u32 = 100,
    max_sequence_length: u32 = 32768,

    const Backend = enum {
        cpu,
        metal,
        cuda,
        webgpu,
    };
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const config = try parseArgs(allocator);

    // Initialize the selected backend
    var backend = try initBackend(allocator, config.backend);
    defer backend.deinit();

    // Load the model
    var model = if (config.model_path) |path|
        try deepseek_core.Model.loadFromPath(allocator, path, backend)
    else
        try deepseek_core.Model.loadDefault(allocator, backend);
    defer model.deinit();

    print("🚀 DeepZig V3 Server Starting...\n", .{});
    print("   Backend: {s}\n", .{@tagName(config.backend)});
    print("   Host: {s}:{d}\n", .{ config.host, config.port });
    print("   Model: {s}\n", .{model.info().name});
    print("   Max Context: {} tokens\n", .{config.max_sequence_length});

    // Start the web server
    var server = try web_layer.Server.init(allocator, .{
        .host = config.host,
        .port = config.port,
        .model = model,
        .max_concurrent_requests = config.max_concurrent_requests,
    });
    defer server.deinit();

    print("✅ Server ready! Send requests to http://{s}:{d}\n", .{ config.host, config.port });
    print("   Endpoints:\n", .{});
    print("   - POST /v1/chat/completions (OpenAI compatible)\n", .{});
    print("   - POST /v1/completions\n", .{});
    print("   - GET  /v1/models\n", .{});
    print("   - GET  /health\n", .{});
    print("   - WebSocket /ws (streaming)\n", .{});

    try server.listen();
}

fn parseArgs(allocator: Allocator) !Config {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var config = Config{};

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--port") and i + 1 < args.len) {
            config.port = try std.fmt.parseInt(u16, args[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--host") and i + 1 < args.len) {
            config.host = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, arg, "--model") and i + 1 < args.len) {
            config.model_path = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, arg, "--backend") and i + 1 < args.len) {
            const backend_str = args[i + 1];
            config.backend = std.meta.stringToEnum(Config.Backend, backend_str) orelse {
                print("Unknown backend: {s}\n", .{backend_str});
                print("Available backends: cpu, metal, cuda, webgpu\n", .{});
                std.process.exit(1);
            };
            i += 1;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        }
    }

    return config;
}

fn initBackend(allocator: Allocator, backend_type: Config.Backend) !deepseek_core.Backend {
    return switch (backend_type) {
        .cpu => cpu_backend.init(allocator),
        .metal => metal_backend.init(allocator),
        .cuda => {
            print("CUDA backend not yet implemented, falling back to CPU\n", .{});
            return cpu_backend.init(allocator);
        },
        .webgpu => {
            print("WebGPU backend not yet implemented, falling back to CPU\n", .{});
            return cpu_backend.init(allocator);
        },
    };
}

fn printHelp() void {
    print("DeepZig V3 - High-Performance LLM Inference\n\n", .{});
    print("Usage: deepseek-v3-zig [OPTIONS]\n\n", .{});
    print("Options:\n", .{});
    print("  --port <PORT>        Port to listen on (default: 8080)\n", .{});
    print("  --host <HOST>        Host to bind to (default: 127.0.0.1)\n", .{});
    print("  --model <PATH>       Path to model weights\n", .{});
    print("  --backend <BACKEND>  Backend to use: cpu, metal, cuda, webgpu (default: cpu)\n", .{});
    print("  --help, -h           Show this help message\n\n", .{});
    print("Examples:\n", .{});
    print("  deepseek-v3-zig --port 3000 --backend metal\n", .{});
    print("  deepseek-v3-zig --model ./models/deepseek-v3.bin --backend cuda\n", .{});
}
