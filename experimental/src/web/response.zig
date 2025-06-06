const std = @import("std");
const http = std.http;
const Allocator = std.mem.Allocator;

/// Response wrapper for easier handling
pub const Response = struct {
    inner: *http.Server.Response,
    allocator: Allocator,
    
    const Self = @This();
    
    pub fn init(inner: *http.Server.Response, allocator: Allocator) Self {
        return Self{
            .inner = inner,
            .allocator = allocator,
        };
    }
    
    /// Set response status
    pub fn setStatus(self: *Self, status: http.Status) void {
        self.inner.status = status;
    }
    
    /// Set header
    pub fn setHeader(self: *Self, name: []const u8, value: []const u8) !void {
        try self.inner.headers.append(name, value);
    }
    
    /// Send JSON response
    pub fn sendJson(self: *Self, data: anytype) !void {
        const json_string = try std.json.stringifyAlloc(
            self.allocator,
            data,
            .{ .whitespace = .indent_2 },
        );
        defer self.allocator.free(json_string);
        
        try self.setHeader("Content-Type", "application/json");
        self.inner.transfer_encoding = .{ .content_length = json_string.len };
        try self.inner.do();
        
        try self.inner.writeAll(json_string);
        try self.inner.finish();
    }
    
    /// Send text response
    pub fn sendText(self: *Self, text: []const u8) !void {
        try self.setHeader("Content-Type", "text/plain");
        self.inner.transfer_encoding = .{ .content_length = text.len };
        try self.inner.do();
        
        try self.inner.writeAll(text);
        try self.inner.finish();
    }
    
    /// Send HTML response
    pub fn sendHtml(self: *Self, html: []const u8) !void {
        try self.setHeader("Content-Type", "text/html");
        self.inner.transfer_encoding = .{ .content_length = html.len };
        try self.inner.do();
        
        try self.inner.writeAll(html);
        try self.inner.finish();
    }
    
    /// Send error response
    pub fn sendError(self: *Self, status: http.Status, message: []const u8) !void {
        const error_response = struct {
            @"error": struct {
                message: []const u8,
                type: []const u8,
                code: u16,
            },
        }{
            .@"error" = .{
                .message = message,
                .type = "error",
                .code = @intFromEnum(status),
            },
        };
        
        self.setStatus(status);
        try self.sendJson(error_response);
    }
    
    /// Redirect to another URL
    pub fn redirect(self: *Self, location: []const u8) !void {
        self.setStatus(.found);
        try self.setHeader("Location", location);
        try self.sendText("");
    }
}; 