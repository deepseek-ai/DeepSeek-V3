// WebAssembly Entry Point for DeepSeek V3
// Enables browser-based inference with minimal dependencies

const std = @import("std");
const deepseek_core = @import("deepseek_core");

// WebAssembly allocator using the heap
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// WebAssembly exports for JavaScript interop
/// These functions are callable from JavaScript

/// Initialize the model (exported to JS)
export fn wasm_init_model() i32 {
    // TODO: Initialize a smaller model suitable for browser
    std.log.info("Initializing DeepSeek V3 for WebAssembly", .{});
    
    // For browser use, we'd use a much smaller model or quantized version
    // Return success status
    return 0; // Success
}

/// Generate text completion (exported to JS)
export fn wasm_generate_text(
    input_ptr: [*]const u8,
    input_len: u32,
    output_ptr: [*]u8,
    output_max_len: u32,
) u32 {
    const input = input_ptr[0..input_len];
    const output_buffer = output_ptr[0..output_max_len];
    
    std.log.info("WASM text generation: {s}", .{input});
    
    // TODO: Implement actual generation
    // For now, return a placeholder response
    const response = "Hello from DeepSeek V3 WASM! Input was: ";
    const full_response = std.fmt.bufPrint(
        output_buffer, 
        "{s}{s}", 
        .{ response, input }
    ) catch {
        // If buffer too small, return error length
        return 0;
    };
    
    return @intCast(full_response.len);
}

/// Tokenize text (exported to JS)
export fn wasm_tokenize(
    text_ptr: [*]const u8,
    text_len: u32,
    tokens_ptr: [*]u32,
    max_tokens: u32,
) u32 {
    const text = text_ptr[0..text_len];
    const tokens_buffer = tokens_ptr[0..max_tokens];
    
    // TODO: Implement actual tokenization
    // For now, return dummy tokens
    const token_count = @min(text.len / 4, max_tokens); // Rough estimate
    
    for (0..token_count) |i| {
        tokens_buffer[i] = @intCast(i + 1000); // Dummy token IDs
    }
    
    return @intCast(token_count);
}

/// Get model information (exported to JS)
export fn wasm_get_model_info(
    info_ptr: [*]u8,
    info_max_len: u32,
) u32 {
    const info_buffer = info_ptr[0..info_max_len];
    
    const model_info = 
        \\{"name":"DeepSeek-V3-WASM","version":"0.1.0","context_length":4096}
    ;
    
    if (model_info.len > info_max_len) {
        return 0; // Buffer too small
    }
    
    @memcpy(info_buffer[0..model_info.len], model_info);
    return @intCast(model_info.len);
}

/// Allocate memory for JavaScript (exported to JS)
export fn wasm_alloc(size: u32) ?*anyopaque {
    const bytes = allocator.alloc(u8, size) catch return null;
    return bytes.ptr;
}

/// Free memory allocated by wasm_alloc (exported to JS)
export fn wasm_free(ptr: ?*anyopaque, size: u32) void {
    if (ptr) |p| {
        const bytes: [*]u8 = @ptrCast(p);
        allocator.free(bytes[0..size]);
    }
}

/// Main entry point (called by Zig, not exported to JS)
pub fn main() !void {
    std.log.info("DeepSeek V3 WebAssembly module loaded", .{});
    
    // Initialize core components
    deepseek_core.init();
    
    // WASM modules don't have a traditional main loop
    // All interaction happens through exported functions
}

/// Panic handler for WebAssembly
pub fn panic(message: []const u8, stack_trace: ?*std.builtin.StackTrace, ret_addr: ?usize) noreturn {
    _ = stack_trace;
    _ = ret_addr;
    
    // In WASM, we can't print to stderr normally
    // Log the panic message and abort
    std.log.err("WASM Panic: {s}", .{message});
    
    // Trap the WebAssembly execution
    unreachable;
}