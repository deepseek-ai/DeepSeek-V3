// DeepSeek V3 Web Layer
// HTTP server and API endpoints

const std = @import("std");

// Web components
pub const Server = @import("server.zig").Server;
pub const handlers = @import("handlers.zig");
pub const middleware = @import("middleware.zig");
pub const websocket = @import("websocket.zig");

// OpenAI API compatibility
pub const openai = @import("openai.zig");

// Response types
pub const Response = @import("response.zig").Response;
pub const Request = @import("request.zig").Request;

// Error handling
pub const WebError = error{
    InvalidRequest,
    Unauthorized,
    RateLimited,
    ServerError,
    ModelNotFound,
    BadRequest,
};

// Tests
test "web layer" {
    const testing = std.testing;
    _ = testing;
    // TODO: Add web layer tests
} 