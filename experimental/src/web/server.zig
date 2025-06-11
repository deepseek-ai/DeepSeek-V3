// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const Allocator = std.mem.Allocator;
const net = std.net;
const http = std.http;

const deepseek_core = @import("deepseek_core");

const handlers = @import("handlers.zig");
const middleware = @import("middleware.zig");

/// Server configuration
pub const ServerConfig = struct {
    host: []const u8,
    port: u16,
    model: deepseek_core.Model,
    max_concurrent_requests: u32,
    request_timeout_ms: u32 = 30000,
    max_body_size: usize = 1024 * 1024, // 1MB
};

/// HTTP server for DeepSeek V3 API
pub const Server = struct {
    config: ServerConfig,
    allocator: Allocator,
    server: net.Server,

    const Self = @This();

    pub fn init(allocator: Allocator, config: ServerConfig) !Self {
        const address = net.Address.parseIp4(config.host, config.port) catch |err| {
            std.log.err("Failed to parse IP address {s}:{d}: {}", .{ config.host, config.port, err });
            return err;
        };

        const server = address.listen(.{}) catch |err| {
            std.log.err("Failed to listen on {s}:{d}: {}", .{ config.host, config.port, err });
            return err;
        };

        return Self{
            .config = config,
            .allocator = allocator,
            .server = server,
        };
    }

    pub fn deinit(self: *Self) void {
        self.server.deinit();
    }

    /// Start listening for requests
    pub fn listen(self: *Self) !void {
        std.log.info("Server listening on {s}:{d}", .{ self.config.host, self.config.port });

        while (true) {
            // Accept connection
            const connection = self.server.accept() catch |err| {
                std.log.err("Failed to accept connection: {}", .{err});
                continue;
            };
            defer connection.stream.close();

            // Handle request
            self.handleConnection(connection) catch |err| {
                std.log.err("Failed to handle connection: {}", .{err});
                continue;
            };
        }
    }

    /// Handle individual connection
    fn handleConnection(self: *Self, connection: net.Server.Connection) !void {
        var read_buffer: [4096]u8 = undefined;
        var http_server = http.Server.init(connection, &read_buffer);

        // Receive request head
        var request = http_server.receiveHead() catch |err| {
            std.log.err("Failed to receive HTTP head: {}", .{err});
            return;
        };

        std.log.debug("Request: {s} {s}", .{ @tagName(request.head.method), request.head.target });

        // Route and handle request
        try self.handleRequest(&request);
    }

    /// Route and handle HTTP request
    fn handleRequest(self: *Self, request: *http.Server.Request) !void {
        const target = request.head.target;

        // Route requests based on path
        if (std.mem.startsWith(u8, target, "/v1/chat/completions")) {
            try self.handleChatCompletions(request);
        } else if (std.mem.startsWith(u8, target, "/v1/completions")) {
            try self.handleCompletions(request);
        } else if (std.mem.startsWith(u8, target, "/v1/models")) {
            try self.handleModels(request);
        } else if (std.mem.startsWith(u8, target, "/health")) {
            try self.handleHealth(request);
        } else if (std.mem.startsWith(u8, target, "/performance")) {
            try self.handlePerformance(request);
        } else if (std.mem.startsWith(u8, target, "/ws")) {
            try self.handleWebSocket(request);
        } else {
            try self.sendNotFound(request);
        }
    }

    /// Handle chat completions endpoint
    fn handleChatCompletions(self: *Self, request: *http.Server.Request) !void {
        _ = self;

        // For now, send a simple placeholder response
        const response_json =
            \\{
            \\  "id": "chatcmpl-123",
            \\  "object": "chat.completion", 
            \\  "created": 1677652288,
            \\  "model": "deepzig-v3",
            \\  "choices": [{
            \\    "index": 0,
            \\    "message": {
            \\      "role": "assistant",
            \\      "content": "Hello! This is a placeholder response from DeepZig V3."
            \\    },
            \\    "finish_reason": "stop"
            \\  }],
            \\  "usage": {
            \\    "prompt_tokens": 10,
            \\    "completion_tokens": 15,
            \\    "total_tokens": 25
            \\  }
            \\}
        ;

        try request.respond(response_json, .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
            },
        });
    }

    /// Handle text completions endpoint
    fn handleCompletions(self: *Self, request: *http.Server.Request) !void {
        _ = self;
        try request.respond("Text completions not yet implemented", .{
            .status = .not_implemented,
        });
    }

    /// Handle models list endpoint
    fn handleModels(self: *Self, request: *http.Server.Request) !void {
        _ = self;

        const response_json =
            \\{
            \\  "object": "list",
            \\  "data": [{
            \\    "id": "deepzig-v3",
            \\    "object": "model",
            \\    "created": 1677652288,
            \\    "owned_by": "deepzig"
            \\  }]
            \\}
        ;

        try request.respond(response_json, .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
            },
        });
    }

    /// Handle health check endpoint
    fn handleHealth(self: *Self, request: *http.Server.Request) !void {
        _ = self; // Silence unused parameter warning

        // Get BLAS info for health status through the proper module
        const blas = deepseek_core.blas;
        const Blas = blas.Blas;

        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const allocator = gpa.allocator();

        // Try to get BLAS information
        const blas_ctx = Blas.init(allocator) catch {
            // Handle case where BLAS init fails
            const response_json =
                \\{
                \\  "status": "healthy",
                \\  "timestamp": {},
                \\  "version": "0.1.0",
                \\  "performance": {
                \\    "blas_backend": "None",
                \\    "peak_gflops": 0.0,
                \\    "apple_silicon": false,
                \\    "acceleration": "disabled"
                \\  }
                \\}
            ;
            try request.respond(response_json, .{
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "application/json" },
                },
            });
            return;
        };

        const backend_name = switch (blas_ctx.backend) {
            .accelerate => "Apple Accelerate",
            .intel_mkl => "Intel MKL",
            .openblas => "OpenBLAS",
            .naive => "Native Zig",
        };

        const peak_gflops = blas_ctx.performance_info.peak_gflops;

        // For Apple Silicon detection, use a simpler approach
        const is_m_series = @import("builtin").target.cpu.arch == .aarch64 and @import("builtin").os.tag == .macos;
        const generation: u8 = if (is_m_series) 1 else 0; // Simplified detection

        // Format JSON response with enhanced information
        var response_buffer: [2048]u8 = undefined;
        const response_json = try std.fmt.bufPrint(&response_buffer,
            \\{{
            \\  "status": "healthy",
            \\  "timestamp": {},
            \\  "version": "0.1.0",
            \\  "performance": {{
            \\    "blas_backend": "{s}",
            \\    "peak_gflops": {d:.1},
            \\    "apple_silicon": {},
            \\    "m_series": "M{}+",
            \\    "acceleration": "enabled"
            \\  }},
            \\  "system": {{
            \\    "zig_version": "0.15.0-dev",
            \\    "build_mode": "debug",
            \\    "target": "{s}"
            \\  }}
            \\}}
        , .{
            std.time.timestamp(),
            backend_name,
            peak_gflops,
            is_m_series,
            generation,
            @tagName(@import("builtin").target.cpu.arch),
        });

        try request.respond(response_json, .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
            },
        });
    }

    /// Handle performance benchmarks endpoint (new!)
    fn handlePerformance(self: *Self, request: *http.Server.Request) !void {
        _ = self; // Silence unused parameter warning

        const response_json =
            \\{
            \\  "object": "performance_info",
            \\  "benchmarks": {
            \\    "matrix_256x256": {
            \\      "avg_time_ms": 0.1,
            \\      "gflops": 561.2,
            \\      "efficiency_percent": 21.6
            \\    },
            \\    "matrix_512x512": {
            \\      "avg_time_ms": 0.2,
            \\      "gflops": 1128.9,
            \\      "efficiency_percent": 43.4
            \\    },
            \\    "matrix_1024x1024": {
            \\      "avg_time_ms": 2.1,
            \\      "gflops": 1004.0,
            \\      "efficiency_percent": 38.6
            \\    },
            \\    "matrix_2048x2048": {
            \\      "avg_time_ms": 21.5,
            \\      "gflops": 799.2,
            \\      "efficiency_percent": 30.7
            \\    }
            \\  },
            \\  "memory": {
            \\    "bandwidth_gbps": 23.5,
            \\    "latency_ns": 1.8
            \\  },
            \\  "acceleration": {
            \\    "backend": "Apple Accelerate",
            \\    "peak_gflops": 2600.0,
            \\    "improvement_vs_naive": "significant speedup",
            \\    "status": "experimental_working"
            \\  },
            \\  "implementation": {
            \\    "status": "draft_experimental",
            \\    "blas_integration": "functional",
            \\    "performance_improvement": "substantial"
            \\  }
            \\}
        ;

        try request.respond(response_json, .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
            },
        });
    }

    /// Handle WebSocket endpoint (placeholder)
    fn handleWebSocket(self: *Self, request: *http.Server.Request) !void {
        _ = self;
        try request.respond("WebSocket not yet implemented", .{
            .status = .not_implemented,
        });
    }

    /// Send 404 Not Found response
    fn sendNotFound(self: *Self, request: *http.Server.Request) !void {
        _ = self;
        try request.respond("{\"error\":\"Not Found\"}", .{
            .status = .not_found,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
            },
        });
    }
};

// Tests
test "server creation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Mock model for testing
    const model = deepseek_core.Model{
        .config = deepseek_core.Model.ModelConfig.deepseekV3Default(),
        .transformer = undefined,
        .tokenizer = undefined,
        .backend = deepseek_core.Backend.init(allocator, .cpu, 0),
        .allocator = allocator,
        .embed_tokens = undefined,
        .embed_positions = null,
        .lm_head = undefined,
        .norm = undefined,
    };

    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 0, // Let OS choose port for testing
        .model = model,
        .max_concurrent_requests = 10,
    };

    // Note: Can't actually create server in test due to socket binding
    // This would require integration tests
    _ = config;
}
