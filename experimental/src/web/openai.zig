const std = @import("std");

// OpenAI API compatible structures

/// Chat completion request
pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []Message,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    stream: ?bool = null,
};

/// Chat message
pub const Message = struct {
    role: []const u8, // "system", "user", "assistant"
    content: []const u8,
};

/// Chat completion response
pub const ChatCompletionResponse = struct {
    id: []const u8,
    object: []const u8, // "chat.completion"
    created: i64,
    model: []const u8,
    choices: []Choice,
    usage: Usage,
};

/// Choice in completion response
pub const Choice = struct {
    index: u32,
    message: Message,
    finish_reason: []const u8, // "stop", "length", "content_filter"
};

/// Token usage information
pub const Usage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
};

/// Models list response
pub const ModelsResponse = struct {
    object: []const u8, // "list"
    data: []ModelInfo,
};

/// Model information
pub const ModelInfo = struct {
    id: []const u8,
    object: []const u8, // "model"
    created: i64,
    owned_by: []const u8,
};