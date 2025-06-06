const std = @import("std");
const http = std.http;
const Allocator = std.mem.Allocator;

/// CORS middleware configuration
pub const CorsConfig = struct {
    allow_origins: []const []const u8 = &[_][]const u8{"*"},
    allow_methods: []const []const u8 = &[_][]const u8{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
    allow_headers: []const []const u8 = &[_][]const u8{"Content-Type", "Authorization"},
    max_age: u32 = 86400, // 24 hours
};

/// Add CORS headers to response
pub fn cors(response: *http.Server.Response, config: CorsConfig) !void {
    _ = config;
    // TODO: For now, just add basic CORS headers
    // In a real implementation, you'd check the request origin against allowed origins
    try response.headers.append("Access-Control-Allow-Origin", "*");
    try response.headers.append("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    try response.headers.append("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

/// Request logging middleware
pub fn logRequest(response: *http.Server.Response) void {
    const method = response.request.method;
    const target = response.request.target;
    const timestamp = std.time.timestamp();
    
    std.log.info("[{}] {s} {s}", .{ timestamp, @tagName(method), target });
}

/// Rate limiting middleware (basic implementation)
pub const RateLimiter = struct {
    requests: std.HashMap(u32, RequestCount, std.hash_map.DefaultContext(u32), std.hash_map.default_max_load_percentage),
    allocator: Allocator,
    max_requests: u32,
    window_seconds: u32,
    
    const RequestCount = struct {
        count: u32,
        window_start: i64,
    };
    
    pub fn init(allocator: Allocator, max_requests: u32, window_seconds: u32) RateLimiter {
        return RateLimiter{
            .requests = std.HashMap(u32, RequestCount, std.hash_map.DefaultContext(u32), std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
            .max_requests = max_requests,
            .window_seconds = window_seconds,
        };
    }
    
    pub fn deinit(self: *RateLimiter) void {
        self.requests.deinit();
    }
    
    /// Check if request is allowed (simplified IP-based rate limiting)
    pub fn checkRate(self: *RateLimiter, client_ip: u32) bool {
        const now = std.time.timestamp();
        const window_start = now - self.window_seconds;
        
        const result = self.requests.getOrPut(client_ip) catch return false;
        
        if (!result.found_existing) {
            // New client
            result.value_ptr.* = RequestCount{
                .count = 1,
                .window_start = now,
            };
            return true;
        }
        
        // Check if we're in a new window
        if (result.value_ptr.window_start < window_start) {
            result.value_ptr.count = 1;
            result.value_ptr.window_start = now;
            return true;
        }
        
        // Check if under limit
        if (result.value_ptr.count < self.max_requests) {
            result.value_ptr.count += 1;
            return true;
        }
        
        return false; // Rate limited
    }
};

/// Authentication middleware (basic bearer token)
pub fn authenticateBearer(response: *http.Server.Response, expected_token: []const u8) bool {
    const auth_header = response.request.headers.getFirstValue("Authorization") orelse return false;
    
    if (!std.mem.startsWith(u8, auth_header, "Bearer ")) {
        return false;
    }
    
    const token = auth_header[7..]; // Skip "Bearer "
    return std.mem.eql(u8, token, expected_token);
} 