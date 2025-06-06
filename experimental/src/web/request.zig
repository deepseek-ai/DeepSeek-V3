const std = @import("std");
const http = std.http;
const Allocator = std.mem.Allocator;

/// Request wrapper for easier handling
pub const Request = struct {
    inner: *http.Server.Request,
    allocator: Allocator,
    
    const Self = @This();
    
    pub fn init(inner: *http.Server.Request, allocator: Allocator) Self {
        return Self{
            .inner = inner,
            .allocator = allocator,
        };
    }
    
    /// Get request method
    pub fn method(self: *const Self) http.Method {
        return self.inner.method;
    }
    
    /// Get request path/target
    pub fn path(self: *const Self) []const u8 {
        return self.inner.target;
    }
    
    /// Get header value
    pub fn header(self: *const Self, name: []const u8) ?[]const u8 {
        return self.inner.headers.getFirstValue(name);
    }
    
    /// Get query parameter (simple implementation)
    pub fn query(self: *const Self, name: []const u8) ?[]const u8 {
        const target = self.inner.target;
        if (std.mem.indexOf(u8, target, "?")) |query_start| {
            const query_string = target[query_start + 1..];
            var iter = std.mem.split(u8, query_string, "&");
            
            while (iter.next()) |param| {
                if (std.mem.indexOf(u8, param, "=")) |eq_pos| {
                    const key = param[0..eq_pos];
                    const value = param[eq_pos + 1..];
                    if (std.mem.eql(u8, key, name)) {
                        return value;
                    }
                }
            }
        }
        return null;
    }
    
    /// Extract path parameter (e.g., /users/{id} -> id value)
    pub fn pathParam(self: *const Self, name: []const u8) ?[]const u8 {
        // TODO: Implement proper path parameter extraction
        // This would require route pattern matching
        _ = self;
        _ = name;
        return null;
    }
    
    /// Get content type
    pub fn contentType(self: *const Self) ?[]const u8 {
        return self.header("Content-Type");
    }
    
    /// Check if request is JSON
    pub fn isJson(self: *const Self) bool {
        if (self.contentType()) |ct| {
            return std.mem.startsWith(u8, ct, "application/json");
        }
        return false;
    }
}; 