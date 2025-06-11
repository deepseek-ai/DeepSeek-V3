// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const Allocator = std.mem.Allocator;

const attention = @import("attention.zig");
const Backend = @import("backend.zig").Backend;
const FloatTensor = @import("tensor.zig").FloatTensor;
const model = @import("model.zig");
const moe = @import("moe.zig");

/// RMS Layer Normalization
const RMSNorm = struct {
    weight: FloatTensor,
    eps: f32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, hidden_size: u32, eps: f32) !Self {
        var weight = try FloatTensor.init(allocator, &[_]usize{hidden_size});
        weight.fill(1.0); // Initialize with ones

        return Self{
            .weight = weight,
            .eps = eps,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.weight.deinit();
    }

    pub fn forward(self: *const Self, input: *const FloatTensor, output: *FloatTensor) !void {
        const batch_size = input.shape.dims[0];
        const seq_len = input.shape.dims[1];
        const hidden_size = input.shape.dims[2];

        // RMS normalization: x / rms(x) * weight
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                // Compute RMS
                var sum_squares: f32 = 0.0;
                for (0..hidden_size) |h| {
                    const idx = (b * seq_len + s) * hidden_size + h;
                    const val = input.data[idx];
                    sum_squares += val * val;
                }
                const rms = std.math.sqrt(sum_squares / @as(f32, @floatFromInt(hidden_size)) + self.eps);

                // Apply normalization
                for (0..hidden_size) |h| {
                    const idx = (b * seq_len + s) * hidden_size + h;
                    output.data[idx] = (input.data[idx] / rms) * self.weight.data[h];
                }
            }
        }
    }
};

/// SwiGLU Activation Function (DeepSeek V3 uses SwiGLU)
const SwiGLU = struct {
    gate_proj: FloatTensor,
    up_proj: FloatTensor,
    down_proj: FloatTensor,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, hidden_size: u32, intermediate_size: u32) !Self {
        var gate_proj = try FloatTensor.init(allocator, &[_]usize{ hidden_size, intermediate_size });
        var up_proj = try FloatTensor.init(allocator, &[_]usize{ hidden_size, intermediate_size });
        var down_proj = try FloatTensor.init(allocator, &[_]usize{ intermediate_size, hidden_size });

        // Initialize with Xavier/Glorot
        initializeLinear(&gate_proj);
        initializeLinear(&up_proj);
        initializeLinear(&down_proj);

        return Self{
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.gate_proj.deinit();
        self.up_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: *Self, input: *const FloatTensor, output: *FloatTensor) !void {
        const batch_size = input.shape.dims[0];
        const seq_len = input.shape.dims[1];
        const hidden_size = input.shape.dims[2];
        const intermediate_size = self.gate_proj.shape.dims[1];

        // Reshape input for matrix multiplication
        var input_reshaped = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, hidden_size });
        defer input_reshaped.deinit();
        @memcpy(input_reshaped.data, input.data);

        // Gate projection: gate = input @ gate_proj
        var gate = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, intermediate_size });
        defer gate.deinit();
        try input_reshaped.matmul(&self.gate_proj, &gate);

        // Up projection: up = input @ up_proj
        var up = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, intermediate_size });
        defer up.deinit();
        try input_reshaped.matmul(&self.up_proj, &up);

        // Apply SwiGLU: silu(gate) * up
        for (0..gate.data.len) |i| {
            const x = gate.data[i];
            const silu = x / (1.0 + @exp(-x)); // SiLU activation
            gate.data[i] = silu * up.data[i];
        }

        // Down projection: output = gate @ down_proj
        var output_reshaped = try FloatTensor.init(self.allocator, &[_]usize{ batch_size * seq_len, hidden_size });
        defer output_reshaped.deinit();
        try gate.matmul(&self.down_proj, &output_reshaped);

        // Reshape back to original dimensions
        @memcpy(output.data, output_reshaped.data);
    }

    fn initializeLinear(tensor: *FloatTensor) void {
        var rng = std.Random.DefaultPrng.init(std.crypto.random.int(u64));
        const random = rng.random();

        const fan_in = tensor.shape.dims[0];
        const fan_out = tensor.shape.dims[1];
        const limit = std.math.sqrt(6.0 / @as(f32, @floatFromInt(fan_in + fan_out)));

        for (tensor.data) |*val| {
            val.* = (random.float(f32) - 0.5) * 2.0 * limit;
        }
    }
};

/// DeepSeek V3 Transformer Layer
pub const TransformerLayer = struct {
    layer_idx: u32,

    // Attention components
    attention: attention.MultiHeadLatentAttention,
    attention_norm: RMSNorm,

    // Feed-forward components (MoE or dense)
    mlp: ?SwiGLU, // Dense FFN for non-MoE layers
    moe_layer: ?moe.MoE, // MoE layer (for MoE layers)
    mlp_norm: RMSNorm,

    // Configuration
    config: model.ModelConfig,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, layer_idx: u32, config: model.ModelConfig, backend: Backend) !Self {
        std.log.info("🔧 Initializing Transformer Layer {} (MoE: {})", .{ layer_idx, isMoELayer(layer_idx, config) });

        // Initialize attention with MLA configuration
        const mla_config = attention.MLAConfig{
            .hidden_size = config.hidden_size,
            .num_attention_heads = config.num_attention_heads,
            .num_key_value_heads = config.num_key_value_heads,
            .qk_nope_head_dim = config.qk_nope_head_dim,
            .qk_rope_head_dim = config.qk_rope_head_dim,
            .v_head_dim = config.v_head_dim,
            .rope_base = config.qk_rope_base,
            .max_position_embeddings = config.max_position_embeddings,
            .attention_dropout = 0.0,
            .use_flash_attention = false,
        };

        const mla = try attention.MultiHeadLatentAttention.init(allocator, mla_config, backend);
        const attention_norm = try RMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);
        const mlp_norm = try RMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);

        // Initialize MLP components based on whether this is an MoE layer
        var mlp: ?SwiGLU = null;
        var moe_layer: ?moe.MoE = null;

        if (isMoELayer(layer_idx, config)) {
            // This layer uses MoE
            moe_layer = try moe.MoE.init(allocator, config, backend);
        } else {
            // This layer uses dense FFN
            mlp = try SwiGLU.init(allocator, config.hidden_size, config.intermediate_size);
        }

        return Self{
            .layer_idx = layer_idx,
            .attention = mla,
            .attention_norm = attention_norm,
            .mlp = mlp,
            .moe_layer = moe_layer,
            .mlp_norm = mlp_norm,
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.attention.deinit();
        self.attention_norm.deinit();
        if (self.mlp) |*layer| layer.deinit();
        if (self.moe_layer) |*layer| layer.deinit();
        self.mlp_norm.deinit();
    }

    /// Forward pass through transformer layer
    pub fn forward(
        self: *Self,
        hidden_states: *const FloatTensor,
        attention_mask: ?*const FloatTensor,
        position_ids: ?*const FloatTensor,
        past_key_value: ?*attention.KVCache,
        use_cache: bool,
        output: *FloatTensor,
    ) !void {
        const batch_size = hidden_states.shape.dims[0];
        const seq_len = hidden_states.shape.dims[1];
        const hidden_size = hidden_states.shape.dims[2];

        std.log.debug("🚀 Layer {} Forward: batch={}, seq_len={}, hidden_size={}", .{ self.layer_idx, batch_size, seq_len, hidden_size });

        // 1. Attention block with residual connection
        var attention_norm_output = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, seq_len, hidden_size });
        defer attention_norm_output.deinit();

        // Pre-attention LayerNorm
        try self.attention_norm.forward(hidden_states, &attention_norm_output);

        // Multi-Head Latent Attention
        var attention_output = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, seq_len, hidden_size });
        defer attention_output.deinit();

        try self.attention.forward(
            &attention_norm_output,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
            &attention_output,
        );

        // Residual connection
        var residual1 = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, seq_len, hidden_size });
        defer residual1.deinit();

        try addTensors(hidden_states, &attention_output, &residual1);

        // 2. Feed-forward block with residual connection
        var mlp_norm_output = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, seq_len, hidden_size });
        defer mlp_norm_output.deinit();

        // Pre-MLP LayerNorm
        try self.mlp_norm.forward(&residual1, &mlp_norm_output);

        // Feed-forward (MoE or dense)
        var mlp_output = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, seq_len, hidden_size });
        defer mlp_output.deinit();

        if (self.moe_layer) |*moe_instance| {
            try moe_instance.forward(&mlp_norm_output, &mlp_output);
        } else if (self.mlp) |*dense_mlp| {
            try dense_mlp.forward(&mlp_norm_output, &mlp_output);
        } else {
            return error.NoMLPConfigured;
        }

        // Final residual connection
        try addTensors(&residual1, &mlp_output, output);

        std.log.debug("✅ Layer {} Forward completed", .{self.layer_idx});
    }

    /// Determine if a layer should use MoE based on DeepSeek V3 architecture
    fn isMoELayer(layer_idx: u32, config: model.ModelConfig) bool {
        // DeepSeek V3 uses MoE in specific layers (typically not the first and last few layers)
        const num_layers = config.num_hidden_layers;
        const skip_first = 1;
        const skip_last = 1;

        return layer_idx >= skip_first and layer_idx < (num_layers - skip_last);
    }
};

/// DeepSeek V3 Transformer implementation
pub const Transformer = struct {
    config: model.ModelConfig,
    backend: Backend,
    allocator: Allocator,
    layers: []TransformerLayer,

    const Self = @This();

    pub fn init(allocator: Allocator, config: model.ModelConfig, backend: Backend) !Self {
        std.log.info("🏗️ Initializing DeepSeek V3 Transformer with {} layers", .{config.num_hidden_layers});

        // Allocate transformer layers
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);

        // Initialize each layer
        for (layers, 0..) |*layer, i| {
            layer.* = try TransformerLayer.init(allocator, @intCast(i), config, backend);
        }

        std.log.info("✅ Transformer initialization complete");
        std.log.info("  Total layers: {}", .{config.num_hidden_layers});
        std.log.info("  MoE layers: {}", .{countMoELayers(config)});
        std.log.info("  Dense layers: {}", .{config.num_hidden_layers - countMoELayers(config)});

        return Self{
            .config = config,
            .backend = backend,
            .allocator = allocator,
            .layers = layers,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
    }

    /// Forward pass through all transformer layers
    pub fn forward(
        self: *Self,
        hidden_states: *const FloatTensor,
        attention_mask: ?*const FloatTensor,
        position_ids: ?*const FloatTensor,
        past_key_values: ?[]attention.KVCache,
        use_cache: bool,
        output: *FloatTensor,
    ) !void {
        const batch_size = hidden_states.shape.dims[0];
        const seq_len = hidden_states.shape.dims[1];
        const hidden_size = hidden_states.shape.dims[2];

        std.log.debug("🔥 Transformer Forward: {} layers, batch={}, seq_len={}, hidden_size={}", .{ self.layers.len, batch_size, seq_len, hidden_size });

        // Initialize intermediate tensor for layer outputs
        var current_hidden = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, seq_len, hidden_size });
        defer current_hidden.deinit();
        @memcpy(current_hidden.data, hidden_states.data);

        var next_hidden = try FloatTensor.init(self.allocator, &[_]usize{ batch_size, seq_len, hidden_size });
        defer next_hidden.deinit();

        // Pass through each transformer layer
        for (self.layers, 0..) |*layer, i| {
            const past_kv = if (past_key_values) |kvs| &kvs[i] else null;

            try layer.forward(
                &current_hidden,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
                &next_hidden,
            );

            // Swap tensors for next iteration
            std.mem.swap(FloatTensor, &current_hidden, &next_hidden);
        }

        // Copy final output
        @memcpy(output.data, current_hidden.data);

        std.log.debug("✅ Transformer Forward completed successfully");
    }

    /// Count MoE layers in configuration
    fn countMoELayers(config: model.ModelConfig) u32 {
        var count: u32 = 0;
        for (0..config.num_hidden_layers) |i| {
            if (TransformerLayer.isMoELayer(@intCast(i), config)) {
                count += 1;
            }
        }
        return count;
    }
};

/// Helper function to add two tensors element-wise
fn addTensors(a: *const FloatTensor, b: *const FloatTensor, result: *FloatTensor) !void {
    if (a.data.len != b.data.len or a.data.len != result.data.len) {
        return error.TensorSizeMismatch;
    }

    for (a.data, b.data, result.data) |a_val, b_val, *r_val| {
        r_val.* = a_val + b_val;
    }
}

// Tests
test "transformer layer initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = model.ModelConfig.deepseekV3Default();
    const backend = Backend{
        .type = .cpu,
        .device_id = 0,
        .allocator = allocator,
    };

    var layer = try TransformerLayer.init(allocator, 0, config, backend);
    defer layer.deinit();

    try std.testing.expect(layer.layer_idx == 0);
    try std.testing.expect(layer.config.hidden_size == config.hidden_size);
}

test "transformer initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Use smaller config for testing
    var config = model.ModelConfig.deepseekV3Default();
    config.num_hidden_layers = 4; // Reduce for testing

    const backend = Backend{
        .type = .cpu,
        .device_id = 0,
        .allocator = allocator,
    };

    var transformer = try Transformer.init(allocator, config, backend);
    defer transformer.deinit();

    try std.testing.expect(transformer.layers.len == 4);
}
