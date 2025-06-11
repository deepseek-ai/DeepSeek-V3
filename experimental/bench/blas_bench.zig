// BLAS-specific benchmark suite
// Tests pure BLAS performance without tensor overhead

const std = @import("std");
const print = std.debug.print;

const deepseek_core = @import("deepseek_core");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🧮 DeepSeek V3 BLAS Benchmark Suite\n");
    print("=====================================\n\n");

    try deepseek_core.blas.benchmarkBlas(allocator);
}
