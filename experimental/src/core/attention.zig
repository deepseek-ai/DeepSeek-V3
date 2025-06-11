// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

const Backend = @import("backend.zig").Backend;
const blas = @import("blas.zig");
const CoreError = @import("root.zig").CoreError;
const tensor = @import("tensor.zig");
const FloatTensor = tensor.FloatTensor;

pub const AttentionError = CoreError || error{
    InvalidSequenceLength,
    InvalidHeadDimension,
    KVCacheMismatch,
    AttentionComputationFailed,
};

/// RoPE (Rotary Position Encoding) implementation
const RoPE = struct {
    base: f32,
    dim: u32,
    cos_cache: FloatTensor,
    sin_cache: FloatTensor,
    max_seq_len: u32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, dim: u32, base: f32, max_seq_len: u32) !Self {
        // Pre-compute RoPE embeddings for efficiency
        var cos_cache = try FloatTensor.init(allocator, &[_]usize{ max_seq_len, dim });
        var sin_cache = try FloatTensor.init(allocator, &[_]usize{ max_seq_len, dim });

        // Compute frequency values
        for (0..max_seq_len) |pos| {
            for (0..dim / 2) |i| {
                const freq = 1.0 / math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;

                cos_cache.data[pos * dim + 2 * i] = @cos(angle);
                cos_cache.data[pos * dim + 2 * i + 1] = @cos(angle);
                sin_cache.data[pos * dim + 2 * i] = @sin(angle);
                sin_cache.data[pos * dim + 2 * i + 1] = @sin(angle);
            }
        }

        return Self{
            .base = base,
            .dim = dim,
            .cos_cache = cos_cache,
            .sin_cache = sin_cache,
            .max_seq_len = max_seq_len,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.cos_cache.deinit();
        self.sin_cache.deinit();
    }

    /// Apply rotary position encoding to query/key tensors
    pub fn apply(self: *const Self, tensor_data: *FloatTensor, seq_len: u32, start_pos: u32) !void {
        if (seq_len + start_pos > self.max_seq_len) {
            return AttentionError.InvalidSequenceLength;
        }

        const batch_size = tensor_data.shape.dims[0];
        const num_heads = tensor_data.shape.dims[1];
        const head_dim = tensor_data.shape.dims[3];

        if (head_dim != self.dim) {
            return AttentionError.InvalidHeadDimension;
        }

        // Apply RoPE rotation: x_out = x * cos + rotate_half(x) * sin
        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                for (0..seq_len) |s| {
                    const pos = start_pos + s;
                    for (0..head_dim / 2) |i| {
                        const base_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                        const cos_val = self.cos_cache.data[pos * self.dim + 2 * i];
                        const sin_val = self.sin_cache.data[pos * self.dim + 2 * i];

                        const x1 = tensor_data.data[base_idx + 2 * i];
                        const x2 = tensor_data.data[base_idx + 2 * i + 1];

                        tensor_data.data[base_idx + 2 * i] = x1 * cos_val - x2 * sin_val;
                        tensor_data.data[base_idx + 2 * i + 1] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }
    }
};

/// KV Cache for efficient inference
const KVCache = struct {
    k_cache: FloatTensor,
    v_cache: FloatTensor,
    seq_len: u32,
    max_seq_len: u32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, batch_size: u32, num_heads: u32, head_dim: u32, max_seq_len: u32) !Self {
        var k_cache = try FloatTensor.init(allocator, &[_]usize{ batch_size, num_heads, max_seq_len, head_dim });
        var v_cache = try FloatTensor.init(allocator, &[_]usize{ batch_size, num_heads, max_seq_len, head_dim });

        k_cache.fill(0.0);
        v_cache.fill(0.0);

        return Self{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .seq_len = 0,
            .max_seq_len = max_seq_len,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.k_cache.deinit();
        self.v_cache.deinit();
    }

    /// Update cache with new key/value tensors
    pub fn update(self: *Self, new_k: *const FloatTensor, new_v: *const FloatTensor, start_pos: u32) !void {
        const batch_size = new_k.shape.dims[0];
        const num_heads = new_k.shape.dims[1];
        const new_seq_len = new_k.shape.dims[2];
        const head_dim = new_k.shape.dims[3];

        if (start_pos + new_seq_len > self.max_seq_len) {
            return AttentionError.InvalidSequenceLength;
        }

        // Copy new keys and values into cache
        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                for (0..new_seq_len) |s| {
                    for (0..head_dim) |d| {
                        const src_idx = ((b * num_heads + h) * new_seq_len + s) * head_dim + d;
                        const dst_idx = ((b * num_heads + h) * self.max_seq_len + (start_pos + s)) * head_dim + d;

                        self.k_cache.data[dst_idx] = new_k.data[src_idx];
                        self.v_cache.data[dst_idx] = new_v.data[src_idx];
                    }
                }
            }
        }

        self.seq_len = start_pos + new_seq_len;
    }

    /// Get current keys from cache
    pub fn getKeys(self: *const Self, allocator: Allocator) !FloatTensor {
        const batch_size = self.k_cache.shape.dims[0];
        const num_heads = self.k_cache.shape.dims[1];
        const head_dim = self.k_cache.shape.dims[3];

        var result = try FloatTensor.init(allocator, &[_]usize{ batch_size, num_heads, self.seq_len, head_dim });

        // Copy current sequence from cache
        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                for (0..self.seq_len) |s| {
                    for (0..head_dim) |d| {
                        const src_idx = ((b * num_heads + h) * self.max_seq_len + s) * head_dim + d;
                        const dst_idx = ((b * num_heads + h) * self.seq_len + s) * head_dim + d;
                        result.data[dst_idx] = self.k_cache.data[src_idx];
                    }
                }
            }
        }

        return result;
    }

    /// Get current values from cache
    pub fn getValues(self: *const Self, allocator: Allocator) !FloatTensor {
        const batch_size = self.v_cache.shape.dims[0];
        const num_heads = self.v_cache.shape.dims[1];
        const head_dim = self.v_cache.shape.dims[3];

        var result = try FloatTensor.init(allocator, &[_]usize{ batch_size, num_heads, self.seq_len, head_dim });

        // Copy current sequence from cache
        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                for (0..self.seq_len) |s| {
                    for (0..head_dim) |d| {
                        const src_idx = ((b * num_heads + h) * self.max_seq_len + s) * head_dim + d;
                        const dst_idx = ((b * num_heads + h) * self.seq_len + s) * head_dim + d;
                        result.data[dst_idx] = self.v_cache.data[src_idx];
                    }
                }
            }
        }

        return result;
    }
};

/// Multi-Head Latent Attention Configuration
pub const MLAConfig = struct {
    hidden_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    qk_nope_head_dim: u32, // Non-positional encoding dimension
    qk_rope_head_dim: u32, // RoPE dimension
    v_head_dim: u32, // Value head dimension
    rope_base: f32, // RoPE base frequency
    max_position_embeddings: u32,
    attention_dropout: f32,
    use_flash_attention: bool,

    pub fn validate(self: MLAConfig) !void {
        if (self.num_attention_heads == 0) return AttentionError.InvalidHeadDimension;
        if (self.num_key_value_heads == 0) return AttentionError.InvalidHeadDimension;
        if (self.qk_nope_head_dim + self.qk_rope_head_dim == 0) return AttentionError.InvalidHeadDimension;
        if (self.v_head_dim == 0) return AttentionError.InvalidHeadDimension;
    }
};

/// Multi-Head Latent Attention (MLA) implementation
/// This is the key innovation in DeepSeek V3 for efficient attention computation
pub const MultiHeadLatentAttention = struct {
    config: MLAConfig,

    // Linear projection layers
    q_proj: FloatTensor, // Query projection
    k_proj: FloatTensor, // Key projection
    v_proj: FloatTensor, // Value projection
    o_proj: FloatTensor, // Output projection

    // Latent projections (key MLA innovation)
    kv_a_proj_with_mqa: FloatTensor, // Latent KV projection
    kv_a_layernorm: FloatTensor, // LayerNorm for latent space
    kv_b_proj: FloatTensor, // Latent to KV projection

    // RoPE for positional encoding
    rope: RoPE,

    // KV Cache for inference
    kv_cache: ?KVCache,

    allocator: Allocator,
    backend: Backend,

    const Self = @This();

    /// Initialize Multi-Head Latent Attention
    pub fn init(allocator: Allocator, config: MLAConfig, backend: Backend) !Self {
        try config.validate();

        std.log.info("🧠 Initializing Multi-Head Latent Attention (MLA)");
        std.log.info("  Hidden size: {}", .{config.hidden_size});
        std.log.info("  Attention heads: {}", .{config.num_attention_heads});
        std.log.info("  KV heads: {}", .{config.num_key_value_heads});
        std.log.info("  QK nope dim: {}", .{config.qk_nope_head_dim});
        std.log.info("  QK rope dim: {}", .{config.qk_rope_head_dim});
        std.log.info("  V head dim: {}", .{config.v_head_dim});

        // Calculate dimensions
        const total_qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
        const kv_lora_rank = config.hidden_size / 8; // Typical latent dimension

        // Initialize linear projections with proper dimensions
        var q_proj = try FloatTensor.init(allocator, &[_]usize{ config.hidden_size, config.num_attention_heads * total_qk_head_dim });
        var k_proj = try FloatTensor.init(allocator, &[_]usize{ config.hidden_size, config.num_key_value_heads * total_qk_head_dim });
        var v_proj = try FloatTensor.init(allocator, &[_]usize{ config.hidden_size, config.num_key_value_heads * config.v_head_dim });
        var o_proj = try FloatTensor.init(allocator, &[_]usize{ config.num_attention_heads * config.v_head_dim, config.hidden_size });

        // MLA-specific latent projections
        var kv_a_proj_with_mqa = try FloatTensor.init(allocator, &[_]usize{ config.hidden_size, kv_lora_rank + config.num_key_value_heads * config.qk_rope_head_dim });
        var kv_a_layernorm = try FloatTensor.init(allocator, &[_]usize{kv_lora_rank});
        var kv_b_proj = try FloatTensor.init(allocator, &[_]usize{ kv_lora_rank, config.num_key_value_heads * (config.qk_nope_head_dim + config.v_head_dim) });

        // Initialize weights with Xavier/Glorot initialization
        initializeLinearLayer(&q_proj, allocator);
        initializeLinearLayer(&k_proj, allocator);
        initializeLinearLayer(&v_proj, allocator);
        initializeLinearLayer(&o_proj, allocator);
        initializeLinearLayer(&kv_a_proj_with_mqa, allocator);
        initializeLinearLayer(&kv_b_proj, allocator);
        kv_a_layernorm.fill(1.0); // Initialize LayerNorm weights to 1

        // Initialize RoPE
        const rope = try RoPE.init(allocator, config.qk_rope_head_dim, config.rope_base, config.max_position_embeddings);

        return Self{
            .config = config,
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .o_proj = o_proj,
            .kv_a_proj_with_mqa = kv_a_proj_with_mqa,
            .kv_a_layernorm = kv_a_layernorm,
            .kv_b_proj = kv_b_proj,
            .rope = rope,
            .kv_cache = null,
            .allocator = allocator,
            .backend = backend,
        };
    }

    pub fn deinit(self: *Self) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        self.kv_a_proj_with_mqa.deinit();
        self.kv_a_layernorm.deinit();
        self.kv_b_proj.deinit();
        self.rope.deinit();
        if (self.kv_cache) |*cache| cache.deinit();
    }

    /// Initialize KV cache for inference
    pub fn initKVCache(self: *Self, batch_size: u32, max_seq_len: u32) !void {
        const total_qk_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim;

        self.kv_cache = try KVCache.init(self.allocator, batch_size, self.config.num_key_value_heads, total_qk_head_dim, max_seq_len);
    }

    /// Forward pass through Multi-Head Latent Attention
    pub fn forward(
        self: *Self,
        hidden_states: *const FloatTensor,
        attention_mask: ?*const FloatTensor,
        position_ids: ?*const FloatTensor,
        past_key_value: ?*KVCache,
        use_cache: bool,
        output: *FloatTensor,
    ) !void {
        _ = position_ids; // TODO: Implement position_ids usage
        const batch_size = hidden_states.shape.dims[0];
        const seq_len = hidden_states.shape.dims[1];
        const hidden_size = hidden_states.shape.dims[2];

        std.log.debug("🧠 MLA Forward: batch={}, seq_len={}, hidden_size={}", .{ batch_size, seq_len, hidden_size });

        if (hidden_size != self.config.hidden_size) {
            return AttentionError.InvalidHeadDimension;
        }

        // Step 1: Compute queries using BLAS-accelerated matrix multiplication
        const total_qk_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim;
        var queries = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, self.config.num_attention_heads * total_qk_head_dim });
        defer queries.deinit();

        // Reshape hidden_states for matrix multiplication
        var hidden_reshaped = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, hidden_size });
        defer hidden_reshaped.deinit();
        @memcpy(hidden_reshaped.data, hidden_states.data);

        try hidden_reshaped.matmul(&self.q_proj, &queries);

        // Step 2: MLA Key-Value computation (the innovation!)
        // Project to latent space
        const kv_lora_rank = self.config.hidden_size / 8;
        var kv_a = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, kv_lora_rank + self.config.num_key_value_heads * self.config.qk_rope_head_dim });
        defer kv_a.deinit();

        try hidden_reshaped.matmul(&self.kv_a_proj_with_mqa, &kv_a);

        // Apply LayerNorm to latent part
        try applyLayerNorm(&kv_a, &self.kv_a_layernorm, kv_lora_rank);

        // Project back to key-value space
        var latent_part = try sliceTensor(&kv_a, 1, 0, kv_lora_rank);
        defer latent_part.deinit();

        var kv_b = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, self.config.num_key_value_heads * (self.config.qk_nope_head_dim + self.config.v_head_dim) });
        defer kv_b.deinit();

        try latent_part.matmul(&self.kv_b_proj, &kv_b);

        // Step 3: Extract RoPE and non-RoPE parts
        var rope_part = try sliceTensor(&kv_a, 1, kv_lora_rank, kv_lora_rank + self.config.num_key_value_heads * self.config.qk_rope_head_dim);
        defer rope_part.deinit();

        // Step 4: Combine and reshape keys/values
        var keys = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, self.config.num_key_value_heads, seq_len, total_qk_head_dim });
        defer keys.deinit();

        var values = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, self.config.num_key_value_heads, seq_len, self.config.v_head_dim });
        defer values.deinit();

        try combineKVComponents(&kv_b, &rope_part, &keys, &values, self.config);

        // Step 5: Apply RoPE to queries and keys
        var queries_reshaped = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, self.config.num_attention_heads, seq_len, total_qk_head_dim });
        defer queries_reshaped.deinit();
        try reshapeQueriesForAttention(&queries, &queries_reshaped, self.config);

        const start_pos = if (past_key_value) |cache| cache.seq_len else 0;

        // Apply RoPE to RoPE portions only
        try self.rope.apply(&queries_reshaped, @intCast(seq_len), @intCast(start_pos));
        try self.rope.apply(&keys, @intCast(seq_len), @intCast(start_pos));

        // Step 6: Update KV cache if needed
        if (use_cache) {
            if (self.kv_cache) |*cache| {
                try cache.update(&keys, &values, @intCast(start_pos));
            }
        }

        // Step 7: Compute scaled dot-product attention with BLAS
        var attention_output = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, self.config.num_attention_heads, seq_len, self.config.v_head_dim });
        defer attention_output.deinit();

        try scaledDotProductAttention(&queries_reshaped, &keys, &values, attention_mask, &attention_output, self.config);

        // Step 8: Output projection using BLAS
        var attention_flat = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, self.config.num_attention_heads * self.config.v_head_dim });
        defer attention_flat.deinit();
        try flattenAttentionOutput(&attention_output, &attention_flat);

        var output_flat = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, self.config.hidden_size });
        defer output_flat.deinit();

        try attention_flat.matmul(&self.o_proj, &output_flat);

        // Reshape back to original dimensions
        @memcpy(output.data, output_flat.data);

        std.log.debug("✅ MLA Forward completed successfully");
    }
};

// Helper functions for MLA implementation

/// Initialize linear layer with Xavier/Glorot uniform initialization
fn initializeLinearLayer(layer_tensor: *FloatTensor, allocator: Allocator) void {
    _ = allocator;
    var rng = std.Random.DefaultPrng.init(std.crypto.random.int(u64));
    const random = rng.random();

    const fan_in = layer_tensor.shape.dims[0];
    const fan_out = layer_tensor.shape.dims[1];
    const limit = math.sqrt(6.0 / @as(f32, @floatFromInt(fan_in + fan_out)));

    for (layer_tensor.data) |*val| {
        val.* = (random.float(f32) - 0.5) * 2.0 * limit;
    }
}

/// Apply LayerNorm to a portion of the tensor
fn applyLayerNorm(input_tensor: *FloatTensor, norm_weights: *const FloatTensor, latent_dim: u32) !void {
    const batch_seq = input_tensor.shape.dims[0];
    const eps: f32 = 1e-6;

    for (0..batch_seq) |i| {
        // Compute mean and variance for latent portion
        var mean: f32 = 0.0;
        for (0..latent_dim) |j| {
            mean += input_tensor.data[i * input_tensor.shape.dims[1] + j];
        }
        mean /= @floatFromInt(latent_dim);

        var variance: f32 = 0.0;
        for (0..latent_dim) |j| {
            const diff = input_tensor.data[i * input_tensor.shape.dims[1] + j] - mean;
            variance += diff * diff;
        }
        variance /= @floatFromInt(latent_dim);

        // Apply normalization
        const inv_std = 1.0 / math.sqrt(variance + eps);
        for (0..latent_dim) |j| {
            const idx = i * input_tensor.shape.dims[1] + j;
            input_tensor.data[idx] = (input_tensor.data[idx] - mean) * inv_std * norm_weights.data[j];
        }
    }
}

/// Slice a tensor along a specific dimension
fn sliceTensor(input_tensor: *const FloatTensor, dim: u32, start: u32, end: u32) !FloatTensor {
    // Simple implementation for 2D tensors
    if (dim != 1) return error.UnsupportedSliceDimension;

    const rows = input_tensor.shape.dims[0];
    const slice_width = end - start;

    var result = try FloatTensor.init(input_tensor.allocator, &[_]usize{ rows, slice_width });

    for (0..rows) |i| {
        for (0..slice_width) |j| {
            result.data[i * slice_width + j] = input_tensor.data[i * input_tensor.shape.dims[1] + start + j];
        }
    }

    return result;
}

/// Combine KV components from latent space and RoPE components
fn combineKVComponents(
    kv_b: *const FloatTensor,
    rope_part: *const FloatTensor,
    keys: *FloatTensor,
    values: *FloatTensor,
    config: MLAConfig,
) !void {
    const batch_size = keys.shape.dims[0];
    const num_kv_heads = config.num_key_value_heads;
    const seq_len = keys.shape.dims[2];
    const qk_nope_dim = config.qk_nope_head_dim;
    const qk_rope_dim = config.qk_rope_head_dim;
    const v_dim = config.v_head_dim;

    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            const seq_idx = b * seq_len + s;

            for (0..num_kv_heads) |h| {
                // Copy key components (nope + rope)
                for (0..qk_nope_dim) |d| {
                    const src_idx = seq_idx * (num_kv_heads * (qk_nope_dim + v_dim)) + h * (qk_nope_dim + v_dim) + d;
                    const dst_idx = ((b * num_kv_heads + h) * seq_len + s) * (qk_nope_dim + qk_rope_dim) + d;
                    keys.data[dst_idx] = kv_b.data[src_idx];
                }

                for (0..qk_rope_dim) |d| {
                    const src_idx = seq_idx * (num_kv_heads * qk_rope_dim) + h * qk_rope_dim + d;
                    const dst_idx = ((b * num_kv_heads + h) * seq_len + s) * (qk_nope_dim + qk_rope_dim) + qk_nope_dim + d;
                    keys.data[dst_idx] = rope_part.data[src_idx];
                }

                // Copy value components
                for (0..v_dim) |d| {
                    const src_idx = seq_idx * (num_kv_heads * (qk_nope_dim + v_dim)) + h * (qk_nope_dim + v_dim) + qk_nope_dim + d;
                    const dst_idx = ((b * num_kv_heads + h) * seq_len + s) * v_dim + d;
                    values.data[dst_idx] = kv_b.data[src_idx];
                }
            }
        }
    }
}

/// Reshape queries for attention computation
fn reshapeQueriesForAttention(queries: *const FloatTensor, queries_reshaped: *FloatTensor, config: MLAConfig) !void {
    const batch_size = queries_reshaped.shape.dims[0];
    const num_heads = config.num_attention_heads;
    const seq_len = queries_reshaped.shape.dims[2];
    const head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;

    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            for (0..num_heads) |h| {
                for (0..head_dim) |d| {
                    const src_idx = (b * seq_len + s) * (num_heads * head_dim) + h * head_dim + d;
                    const dst_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
                    queries_reshaped.data[dst_idx] = queries.data[src_idx];
                }
            }
        }
    }
}

/// Scaled dot-product attention with BLAS acceleration
fn scaledDotProductAttention(
    queries: *const FloatTensor,
    keys: *const FloatTensor,
    values: *const FloatTensor,
    attention_mask: ?*const FloatTensor,
    output: *FloatTensor,
    config: MLAConfig,
) !void {
    _ = attention_mask; // TODO: Implement attention masking

    const batch_size = queries.shape.dims[0];
    const num_heads = queries.shape.dims[1];
    const seq_len = queries.shape.dims[2];
    const head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    const v_head_dim = config.v_head_dim;

    const scale = 1.0 / math.sqrt(@as(f32, @floatFromInt(head_dim)));

    // For each batch and head, compute attention
    for (0..batch_size) |b| {
        for (0..num_heads) |h| {
            // Extract Q, K, V for this batch/head
            var q_slice = try FloatTensor.init(queries.allocator, &[_]usize{ seq_len, head_dim });
            defer q_slice.deinit();
            var k_slice = try FloatTensor.init(keys.allocator, &[_]usize{ seq_len, head_dim });
            defer k_slice.deinit();
            var v_slice = try FloatTensor.init(values.allocator, &[_]usize{ seq_len, v_head_dim });
            defer v_slice.deinit();

            // Copy data for this batch/head
            for (0..seq_len) |s| {
                for (0..head_dim) |d| {
                    const src_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
                    q_slice.data[s * head_dim + d] = queries.data[src_idx];
                    k_slice.data[s * head_dim + d] = keys.data[src_idx];
                }
                for (0..v_head_dim) |d| {
                    const src_idx = ((b * num_heads + h) * seq_len + s) * v_head_dim + d;
                    v_slice.data[s * v_head_dim + d] = values.data[src_idx];
                }
            }

            // Compute Q @ K^T using BLAS
            var k_transposed = try FloatTensor.init(keys.allocator, &[_]usize{ head_dim, seq_len });
            defer k_transposed.deinit();
            transposeMatrix(&k_slice, &k_transposed);

            var scores = try FloatTensor.init(queries.allocator, &[_]usize{ seq_len, seq_len });
            defer scores.deinit();
            try q_slice.matmul(&k_transposed, &scores);

            // Scale scores
            for (scores.data) |*score| {
                score.* *= scale;
            }

            // Apply softmax
            applySoftmax(&scores);

            // Compute scores @ V using BLAS
            var attention_out = try FloatTensor.init(output.allocator, &[_]usize{ seq_len, v_head_dim });
            defer attention_out.deinit();
            try scores.matmul(&v_slice, &attention_out);

            // Copy back to output
            for (0..seq_len) |s| {
                for (0..v_head_dim) |d| {
                    const dst_idx = ((b * num_heads + h) * seq_len + s) * v_head_dim + d;
                    output.data[dst_idx] = attention_out.data[s * v_head_dim + d];
                }
            }
        }
    }
}

/// Transpose a 2D matrix
fn transposeMatrix(input: *const FloatTensor, output: *FloatTensor) void {
    const rows = input.shape.dims[0];
    const cols = input.shape.dims[1];

    for (0..rows) |i| {
        for (0..cols) |j| {
            output.data[j * rows + i] = input.data[i * cols + j];
        }
    }
}

/// Apply softmax to the last dimension
fn applySoftmax(input_tensor: *FloatTensor) void {
    const rows = input_tensor.shape.dims[0];
    const cols = input_tensor.shape.dims[1];

    for (0..rows) |i| {
        // Find max for numerical stability
        var max_val = input_tensor.data[i * cols];
        for (1..cols) |j| {
            const val = input_tensor.data[i * cols + j];
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        var sum: f32 = 0.0;
        for (0..cols) |j| {
            const val = @exp(input_tensor.data[i * cols + j] - max_val);
            input_tensor.data[i * cols + j] = val;
            sum += val;
        }

        // Normalize
        for (0..cols) |j| {
            input_tensor.data[i * cols + j] /= sum;
        }
    }
}

/// Flatten attention output for final projection
fn flattenAttentionOutput(attention_output: *const FloatTensor, output: *FloatTensor) !void {
    @memcpy(output.data, attention_output.data);
}

// Tests
test "MLA initialization and basic operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = MLAConfig{
        .hidden_size = 768,
        .num_attention_heads = 12,
        .num_key_value_heads = 12,
        .qk_nope_head_dim = 64,
        .qk_rope_head_dim = 32,
        .v_head_dim = 64,
        .rope_base = 10000.0,
        .max_position_embeddings = 2048,
        .attention_dropout = 0.1,
        .use_flash_attention = false,
    };

    const backend = Backend{
        .type = .cpu,
        .device_id = 0,
        .allocator = allocator,
    };

    var mla = try MultiHeadLatentAttention.init(allocator, config, backend);
    defer mla.deinit();

    // Test basic tensor shapes
    try std.testing.expect(mla.q_proj.shape.dims[0] == 768);
    try std.testing.expect(mla.rope.dim == 32);
}

test "RoPE functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rope = try RoPE.init(allocator, 64, 10000.0, 128);
    defer rope.deinit();

    var test_tensor = try FloatTensor.init(allocator, &[_]usize{ 1, 1, 4, 64 });
    defer test_tensor.deinit();
    test_tensor.fillRandom(42);

    try rope.apply(&test_tensor, 4, 0);

    // Just verify it doesn't crash - detailed testing would require reference implementation
}
