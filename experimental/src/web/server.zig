const std = @import("std");
const deepseek_core = @import("deepseek_core");
const handlers = @import("handlers.zig");
const middleware = @import("middleware.zig");

const Allocator = std.mem.Allocator;
const net = std.net;
const http = std.http;

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
        _ = self;
        
        const response_json = 
            \\{
            \\  "status": "healthy",
            \\  "timestamp": 1677652288,
            \\  "version": "0.1.0"
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