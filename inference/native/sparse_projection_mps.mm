#include <torch/extension.h>
#include <torch/csrc/api/include/torch/mps.h>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include <mutex>
#include <tuple>

namespace {

constexpr uint32_t kMaxActive = 512;

struct SparseProjectionParams {
  uint32_t tokens;
  uint32_t dim;
  uint32_t max_active;
  uint32_t out_features;
  uint32_t has_bias;
};

struct SparseProjectionPairParams {
  uint32_t tokens;
  uint32_t dim;
  uint32_t max_active;
  uint32_t out_features_a;
  uint32_t out_features_b;
  uint32_t has_bias_a;
  uint32_t has_bias_b;
};

static const char* kSparseProjectionMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint SPARSE_MAX_ACTIVE = 512;

struct SparseProjectionParams {
    uint tokens;
    uint dim;
    uint max_active;
    uint out_features;
    uint has_bias;
};

struct SparseProjectionPairParams {
    uint tokens;
    uint dim;
    uint max_active;
    uint out_features_a;
    uint out_features_b;
    uint has_bias_a;
    uint has_bias_b;
};

kernel void sparse_projection_kernel(
    device const float* x [[buffer(0)]],
    device const int* active_indices [[buffer(1)]],
    device const int* active_counts [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant SparseProjectionParams& p [[buffer(6)]],
    uint token_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]
) {
    if (token_index >= p.tokens || p.dim == 0 || p.out_features == 0 || p.max_active > SPARSE_MAX_ACTIVE) {
        return;
    }

    threadgroup int shared_indices[SPARSE_MAX_ACTIVE];
    threadgroup float shared_values[SPARSE_MAX_ACTIVE];

    const uint count = min(uint(max(active_counts[token_index], 0)), p.max_active);
    const uint x_base = token_index * p.dim;
    const uint idx_base = token_index * p.max_active;
    const uint out_base = token_index * p.out_features;

    for (uint slot = lane; slot < count; slot += lanes) {
        const int idx = active_indices[idx_base + slot];
        const uint safe_idx = (idx >= 0 && uint(idx) < p.dim) ? uint(idx) : 0u;
        shared_indices[slot] = int(safe_idx);
        shared_values[slot] = x[x_base + safe_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint out_feature = lane; out_feature < p.out_features; out_feature += lanes) {
        float acc = p.has_bias != 0 ? bias[out_feature] : 0.0f;
        const uint weight_base = out_feature * p.dim;
        for (uint slot = 0; slot < count; ++slot) {
            const uint idx = uint(shared_indices[slot]);
            acc += shared_values[slot] * weight[weight_base + idx];
        }
        out[out_base + out_feature] = acc;
    }
}

kernel void sparse_projection_pointwise_kernel(
    device const float* x [[buffer(0)]],
    device const int* active_indices [[buffer(1)]],
    device const int* active_counts [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant SparseProjectionParams& p [[buffer(6)]],
    uint linear_index [[thread_position_in_grid]]
) {
    // Toy-model projections are small enough that the threadgroup setup in the shared
    // kernel can dominate runtime. This pointwise version trades reuse for lower launch
    // overhead and is selected on those smaller shapes.
    const uint total_outputs = p.tokens * p.out_features;
    if (linear_index >= total_outputs || p.dim == 0 || p.out_features == 0 || p.max_active > SPARSE_MAX_ACTIVE) {
        return;
    }

    const uint token_index = linear_index / p.out_features;
    const uint out_feature = linear_index - (token_index * p.out_features);
    const uint count = min(uint(max(active_counts[token_index], 0)), p.max_active);
    const uint x_base = token_index * p.dim;
    const uint idx_base = token_index * p.max_active;
    const uint weight_base = out_feature * p.dim;

    float acc = p.has_bias != 0 ? bias[out_feature] : 0.0f;
    for (uint slot = 0; slot < count; ++slot) {
        const int idx = active_indices[idx_base + slot];
        const uint safe_idx = (idx >= 0 && uint(idx) < p.dim) ? uint(idx) : 0u;
        acc += x[x_base + safe_idx] * weight[weight_base + safe_idx];
    }
    out[linear_index] = acc;
}

kernel void sparse_projection_pair_kernel(
    device const float* x [[buffer(0)]],
    device const int* active_indices [[buffer(1)]],
    device const int* active_counts [[buffer(2)]],
    device const float* weight_a [[buffer(3)]],
    device const float* bias_a [[buffer(4)]],
    device const float* weight_b [[buffer(5)]],
    device const float* bias_b [[buffer(6)]],
    device float* out_a [[buffer(7)]],
    device float* out_b [[buffer(8)]],
    constant SparseProjectionPairParams& p [[buffer(9)]],
    uint token_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]
) {
    // Fusing two projections amortizes the launch and metadata cost over the same gated
    // hidden state. This is the current sparse fast path used by MLA.
    const uint total_outputs = p.out_features_a + p.out_features_b;
    if (token_index >= p.tokens || p.dim == 0 || total_outputs == 0 || p.max_active > SPARSE_MAX_ACTIVE) {
        return;
    }

    threadgroup int shared_indices[SPARSE_MAX_ACTIVE];
    threadgroup float shared_values[SPARSE_MAX_ACTIVE];

    const uint count = min(uint(max(active_counts[token_index], 0)), p.max_active);
    const uint x_base = token_index * p.dim;
    const uint idx_base = token_index * p.max_active;
    const uint out_a_base = token_index * p.out_features_a;
    const uint out_b_base = token_index * p.out_features_b;

    for (uint slot = lane; slot < count; slot += lanes) {
        const int idx = active_indices[idx_base + slot];
        const uint safe_idx = (idx >= 0 && uint(idx) < p.dim) ? uint(idx) : 0u;
        shared_indices[slot] = int(safe_idx);
        shared_values[slot] = x[x_base + safe_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint combined_feature = lane; combined_feature < total_outputs; combined_feature += lanes) {
        const bool is_a = combined_feature < p.out_features_a;
        const uint feature = is_a ? combined_feature : combined_feature - p.out_features_a;
        const device float* weight = is_a ? weight_a : weight_b;
        float acc = 0.0f;
        if (is_a) {
            acc = p.has_bias_a != 0 ? bias_a[feature] : 0.0f;
        } else {
            acc = p.has_bias_b != 0 ? bias_b[feature] : 0.0f;
        }
        const uint weight_base = feature * p.dim;
        for (uint slot = 0; slot < count; ++slot) {
            const uint idx = uint(shared_indices[slot]);
            acc += shared_values[slot] * weight[weight_base + idx];
        }
        if (is_a) {
            out_a[out_a_base + feature] = acc;
        } else {
            out_b[out_b_base + feature] = acc;
        }
    }
}

kernel void sparse_projection_pair_pointwise_kernel(
    device const float* x [[buffer(0)]],
    device const int* active_indices [[buffer(1)]],
    device const int* active_counts [[buffer(2)]],
    device const float* weight_a [[buffer(3)]],
    device const float* bias_a [[buffer(4)]],
    device const float* weight_b [[buffer(5)]],
    device const float* bias_b [[buffer(6)]],
    device float* out_a [[buffer(7)]],
    device float* out_b [[buffer(8)]],
    constant SparseProjectionPairParams& p [[buffer(9)]],
    uint linear_index [[thread_position_in_grid]]
) {
    const uint total_features = p.out_features_a + p.out_features_b;
    const uint total_outputs = p.tokens * total_features;
    if (linear_index >= total_outputs || p.dim == 0 || total_features == 0 || p.max_active > SPARSE_MAX_ACTIVE) {
        return;
    }

    const uint token_index = linear_index / total_features;
    const uint combined_feature = linear_index - (token_index * total_features);
    const bool is_a = combined_feature < p.out_features_a;
    const uint feature = is_a ? combined_feature : combined_feature - p.out_features_a;
    const uint count = min(uint(max(active_counts[token_index], 0)), p.max_active);
    const uint x_base = token_index * p.dim;
    const uint idx_base = token_index * p.max_active;
    const device float* weight = is_a ? weight_a : weight_b;

    float acc = 0.0f;
    if (is_a) {
        acc = p.has_bias_a != 0 ? bias_a[feature] : 0.0f;
    } else {
        acc = p.has_bias_b != 0 ? bias_b[feature] : 0.0f;
    }
    const uint weight_base = feature * p.dim;
    for (uint slot = 0; slot < count; ++slot) {
        const int idx = active_indices[idx_base + slot];
        const uint safe_idx = (idx >= 0 && uint(idx) < p.dim) ? uint(idx) : 0u;
        acc += x[x_base + safe_idx] * weight[weight_base + safe_idx];
    }
    if (is_a) {
        out_a[token_index * p.out_features_a + feature] = acc;
    } else {
        out_b[token_index * p.out_features_b + feature] = acc;
    }
}
)METAL";

struct SparseProjectionPipelines {
  id<MTLComputePipelineState> shared = nil;
  id<MTLComputePipelineState> pointwise = nil;
  id<MTLComputePipelineState> pair_shared = nil;
  id<MTLComputePipelineState> pair_pointwise = nil;
};

const SparseProjectionPipelines& get_sparse_projection_pipelines() {
  static SparseProjectionPipelines pipelines;
  static std::once_flag once;
  std::call_once(once, [] {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(device != nil, "Failed to acquire a Metal device for sparse_projection_mps");

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kSparseProjectionMetalSource];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    TORCH_CHECK(
        library != nil,
        "Failed to compile sparse_projection_mps Metal source: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");

    id<MTLFunction> function = [library newFunctionWithName:@"sparse_projection_kernel"];
    TORCH_CHECK(function != nil, "Failed to create sparse_projection_kernel function");
    id<MTLFunction> pointwise_function = [library newFunctionWithName:@"sparse_projection_pointwise_kernel"];
    TORCH_CHECK(pointwise_function != nil, "Failed to create sparse_projection_pointwise_kernel function");
    id<MTLFunction> pair_function = [library newFunctionWithName:@"sparse_projection_pair_kernel"];
    TORCH_CHECK(pair_function != nil, "Failed to create sparse_projection_pair_kernel function");
    id<MTLFunction> pair_pointwise_function = [library newFunctionWithName:@"sparse_projection_pair_pointwise_kernel"];
    TORCH_CHECK(pair_pointwise_function != nil, "Failed to create sparse_projection_pair_pointwise_kernel function");

    pipelines.shared = [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(
        pipelines.shared != nil,
        "Failed to create sparse_projection_mps compute pipeline: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");

    pipelines.pointwise = [device newComputePipelineStateWithFunction:pointwise_function error:&error];
    TORCH_CHECK(
        pipelines.pointwise != nil,
        "Failed to create sparse_projection_pointwise_mps compute pipeline: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");

    pipelines.pair_shared = [device newComputePipelineStateWithFunction:pair_function error:&error];
    TORCH_CHECK(
        pipelines.pair_shared != nil,
        "Failed to create sparse_projection_pair_mps compute pipeline: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");

    pipelines.pair_pointwise = [device newComputePipelineStateWithFunction:pair_pointwise_function error:&error];
    TORCH_CHECK(
        pipelines.pair_pointwise != nil,
        "Failed to create sparse_projection_pair_pointwise_mps compute pipeline: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
  });
  return pipelines;
}

inline id<MTLBuffer> tensor_buffer(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

inline NSUInteger tensor_offset_bytes(const torch::Tensor& tensor) {
  return static_cast<NSUInteger>(tensor.storage_offset() * tensor.element_size());
}

void check_mps_tensor(const torch::Tensor& tensor, const char* name, c10::ScalarType dtype, int64_t dims) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::MPS, name, " must live on MPS");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " must have the expected dtype");
  TORCH_CHECK(tensor.dim() == dims, name, " must have ", dims, " dimensions");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor sparse_projection_mps(
    torch::Tensor x,
    torch::Tensor active_indices,
    torch::Tensor active_counts,
    torch::Tensor weight,
    torch::Tensor bias) {
  check_mps_tensor(x, "x", torch::kFloat32, 2);
  check_mps_tensor(active_indices, "active_indices", torch::kInt32, 2);
  check_mps_tensor(active_counts, "active_counts", torch::kInt32, 1);
  check_mps_tensor(weight, "weight", torch::kFloat32, 2);

  const bool has_bias = bias.defined() && bias.numel() > 0;
  if (has_bias) {
    check_mps_tensor(bias, "bias", torch::kFloat32, 1);
  }

  const auto tokens = x.size(0);
  const auto dim = x.size(1);
  const auto max_active = active_indices.size(1);
  const auto out_features = weight.size(0);

  TORCH_CHECK(tokens > 0, "sparse_projection_mps requires tokens > 0");
  TORCH_CHECK(dim > 0, "sparse_projection_mps requires dim > 0");
  TORCH_CHECK(out_features > 0, "sparse_projection_mps requires out_features > 0");
  TORCH_CHECK(active_counts.size(0) == tokens, "active_counts shape mismatch");
  TORCH_CHECK(active_indices.size(0) == tokens, "active_indices shape mismatch");
  TORCH_CHECK(weight.size(1) == dim, "weight input dimension mismatch");
  TORCH_CHECK(max_active <= static_cast<int64_t>(kMaxActive), "max_active exceeds native limit");
  if (has_bias) {
    TORCH_CHECK(bias.size(0) == out_features, "bias shape mismatch");
  }

  auto output = torch::empty({tokens, out_features}, x.options());

  SparseProjectionParams params{
      static_cast<uint32_t>(tokens),
      static_cast<uint32_t>(dim),
      static_cast<uint32_t>(max_active),
      static_cast<uint32_t>(out_features),
      static_cast<uint32_t>(has_bias ? 1 : 0),
  };

  const SparseProjectionPipelines& pipelines = get_sparse_projection_pipelines();
  const bool use_pointwise = out_features <= 128 && max_active <= 64;
  id<MTLComputePipelineState> pipeline = use_pointwise ? pipelines.pointwise : pipelines.shared;
  dispatch_queue_t queue = torch::mps::get_dispatch_queue();
  TORCH_CHECK(queue != nullptr, "Failed to acquire the PyTorch MPS dispatch queue");

  dispatch_sync(queue, ^{
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer != nil, "Failed to acquire the current PyTorch MPS command buffer");

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    TORCH_CHECK(encoder != nil, "Failed to create a compute command encoder for sparse_projection_mps");

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:tensor_buffer(x) offset:tensor_offset_bytes(x) atIndex:0];
    [encoder setBuffer:tensor_buffer(active_indices) offset:tensor_offset_bytes(active_indices) atIndex:1];
    [encoder setBuffer:tensor_buffer(active_counts) offset:tensor_offset_bytes(active_counts) atIndex:2];
    [encoder setBuffer:tensor_buffer(weight) offset:tensor_offset_bytes(weight) atIndex:3];
    if (has_bias) {
      [encoder setBuffer:tensor_buffer(bias) offset:tensor_offset_bytes(bias) atIndex:4];
    }
    [encoder setBuffer:tensor_buffer(output) offset:tensor_offset_bytes(output) atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];

    const NSUInteger exec_width = pipeline.threadExecutionWidth;
    if (use_pointwise) {
      const NSUInteger preferred = std::max<NSUInteger>(exec_width, 128);
      const NSUInteger threads_per_group =
          std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, preferred);
      const MTLSize threads = MTLSizeMake(
          static_cast<NSUInteger>(tokens * out_features), 1, 1);
      const MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
      [encoder dispatchThreads:threads threadsPerThreadgroup:group];
    } else {
      const NSUInteger rounded =
          ((static_cast<NSUInteger>(out_features) + exec_width - 1) / exec_width) * exec_width;
      const NSUInteger threads_per_group =
          std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, std::max<NSUInteger>(exec_width, rounded));
      const MTLSize threadgroups = MTLSizeMake(static_cast<NSUInteger>(tokens), 1, 1);
      const MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
      [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:group];
    }
    [encoder endEncoding];
  });

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> sparse_projection_pair_mps(
    torch::Tensor x,
    torch::Tensor active_indices,
    torch::Tensor active_counts,
    torch::Tensor weight_a,
    torch::Tensor bias_a,
    torch::Tensor weight_b,
    torch::Tensor bias_b) {
  check_mps_tensor(x, "x", torch::kFloat32, 2);
  check_mps_tensor(active_indices, "active_indices", torch::kInt32, 2);
  check_mps_tensor(active_counts, "active_counts", torch::kInt32, 1);
  check_mps_tensor(weight_a, "weight_a", torch::kFloat32, 2);
  check_mps_tensor(weight_b, "weight_b", torch::kFloat32, 2);

  const bool has_bias_a = bias_a.defined() && bias_a.numel() > 0;
  const bool has_bias_b = bias_b.defined() && bias_b.numel() > 0;
  if (has_bias_a) {
    check_mps_tensor(bias_a, "bias_a", torch::kFloat32, 1);
  }
  if (has_bias_b) {
    check_mps_tensor(bias_b, "bias_b", torch::kFloat32, 1);
  }

  const auto tokens = x.size(0);
  const auto dim = x.size(1);
  const auto max_active = active_indices.size(1);
  const auto out_features_a = weight_a.size(0);
  const auto out_features_b = weight_b.size(0);

  TORCH_CHECK(tokens > 0, "sparse_projection_pair_mps requires tokens > 0");
  TORCH_CHECK(dim > 0, "sparse_projection_pair_mps requires dim > 0");
  TORCH_CHECK(out_features_a > 0, "sparse_projection_pair_mps requires out_features_a > 0");
  TORCH_CHECK(out_features_b > 0, "sparse_projection_pair_mps requires out_features_b > 0");
  TORCH_CHECK(active_counts.size(0) == tokens, "active_counts shape mismatch");
  TORCH_CHECK(active_indices.size(0) == tokens, "active_indices shape mismatch");
  TORCH_CHECK(weight_a.size(1) == dim, "weight_a input dimension mismatch");
  TORCH_CHECK(weight_b.size(1) == dim, "weight_b input dimension mismatch");
  TORCH_CHECK(max_active <= static_cast<int64_t>(kMaxActive), "max_active exceeds native limit");
  if (has_bias_a) {
    TORCH_CHECK(bias_a.size(0) == out_features_a, "bias_a shape mismatch");
  }
  if (has_bias_b) {
    TORCH_CHECK(bias_b.size(0) == out_features_b, "bias_b shape mismatch");
  }

  auto output_a = torch::empty({tokens, out_features_a}, x.options());
  auto output_b = torch::empty({tokens, out_features_b}, x.options());

  SparseProjectionPairParams params{
      static_cast<uint32_t>(tokens),
      static_cast<uint32_t>(dim),
      static_cast<uint32_t>(max_active),
      static_cast<uint32_t>(out_features_a),
      static_cast<uint32_t>(out_features_b),
      static_cast<uint32_t>(has_bias_a ? 1 : 0),
      static_cast<uint32_t>(has_bias_b ? 1 : 0),
  };

  const SparseProjectionPipelines& pipelines = get_sparse_projection_pipelines();
  const uint32_t total_features = params.out_features_a + params.out_features_b;
  const bool use_pointwise = total_features <= 192 && max_active <= 64;
  id<MTLComputePipelineState> pipeline = use_pointwise ? pipelines.pair_pointwise : pipelines.pair_shared;
  dispatch_queue_t queue = torch::mps::get_dispatch_queue();
  TORCH_CHECK(queue != nullptr, "Failed to acquire the PyTorch MPS dispatch queue");

  dispatch_sync(queue, ^{
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer != nil, "Failed to acquire the current PyTorch MPS command buffer");

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    TORCH_CHECK(encoder != nil, "Failed to create a compute command encoder for sparse_projection_pair_mps");

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:tensor_buffer(x) offset:tensor_offset_bytes(x) atIndex:0];
    [encoder setBuffer:tensor_buffer(active_indices) offset:tensor_offset_bytes(active_indices) atIndex:1];
    [encoder setBuffer:tensor_buffer(active_counts) offset:tensor_offset_bytes(active_counts) atIndex:2];
    [encoder setBuffer:tensor_buffer(weight_a) offset:tensor_offset_bytes(weight_a) atIndex:3];
    if (has_bias_a) {
      [encoder setBuffer:tensor_buffer(bias_a) offset:tensor_offset_bytes(bias_a) atIndex:4];
    }
    [encoder setBuffer:tensor_buffer(weight_b) offset:tensor_offset_bytes(weight_b) atIndex:5];
    if (has_bias_b) {
      [encoder setBuffer:tensor_buffer(bias_b) offset:tensor_offset_bytes(bias_b) atIndex:6];
    }
    [encoder setBuffer:tensor_buffer(output_a) offset:tensor_offset_bytes(output_a) atIndex:7];
    [encoder setBuffer:tensor_buffer(output_b) offset:tensor_offset_bytes(output_b) atIndex:8];
    [encoder setBytes:&params length:sizeof(params) atIndex:9];

    const NSUInteger exec_width = pipeline.threadExecutionWidth;
    if (use_pointwise) {
      const NSUInteger preferred = std::max<NSUInteger>(exec_width, 128);
      const NSUInteger threads_per_group =
          std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, preferred);
      const MTLSize threads =
          MTLSizeMake(static_cast<NSUInteger>(tokens * total_features), 1, 1);
      const MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
      [encoder dispatchThreads:threads threadsPerThreadgroup:group];
    } else {
      const NSUInteger rounded =
          ((static_cast<NSUInteger>(total_features) + exec_width - 1) / exec_width) * exec_width;
      const NSUInteger threads_per_group =
          std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, std::max<NSUInteger>(exec_width, rounded));
      const MTLSize threadgroups = MTLSizeMake(static_cast<NSUInteger>(tokens), 1, 1);
      const MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
      [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:group];
    }
    [encoder endEncoding];
  });

  return std::make_tuple(output_a, output_b);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_projection_mps", &sparse_projection_mps, "Sparse projection on MPS");
  m.def("sparse_projection_pair_mps", &sparse_projection_pair_mps, "Sparse projection pair on MPS");
}
