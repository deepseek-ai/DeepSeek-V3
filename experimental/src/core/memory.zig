const std = @import("std");
const Allocator = std.mem.Allocator;

/// Arena allocator for request-scoped memory
pub const ArenaAllocator = std.heap.ArenaAllocator;

/// Memory pool for tensor allocations
pub const TensorPool = struct {
    allocator: Allocator,
    pool: std.ArrayList([]u8),
    
    pub fn init(allocator: Allocator) TensorPool {
        return TensorPool{
            .allocator = allocator,
            .pool = std.ArrayList([]u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *TensorPool) void {
        for (self.pool.items) |buf| {
            self.allocator.free(buf);
        }
        self.pool.deinit();
    }
    
    pub fn alloc(self: *TensorPool, size: usize) ![]u8 {
        // TODO: Implement memory pooling
        return try self.allocator.alloc(u8, size);
    }
    
    pub fn free(self: *TensorPool, buf: []u8) void {
        // TODO: Return to pool instead of freeing
        self.allocator.free(buf);
    }
}; 