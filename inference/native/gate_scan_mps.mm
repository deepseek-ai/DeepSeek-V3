#include <torch/extension.h>
#include <torch/csrc/api/include/torch/mps.h>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include <mutex>
#include <sstream>

namespace {

constexpr uint32_t kMaxDim = 4096;

struct GateScanParams {
  uint32_t batch_size;
  uint32_t seqlen;
  uint32_t dim;
  uint32_t min_active;
  uint32_t max_active;
  float leak;
  float context_leak;
  float context_scale;
  float threshold_lr;
  float target_rate;
  float trace_decay;
  float active_bonus;
};

static const char* kGateScanMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint GATE_MAX_DIM = 4096;

struct GateScanParams {
    uint batch_size;
    uint seqlen;
    uint dim;
    uint min_active;
    uint max_active;
    float leak;
    float context_leak;
    float context_scale;
    float threshold_lr;
    float target_rate;
    float trace_decay;
    float active_bonus;
};

kernel void gate_scan_kernel(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* context [[buffer(2)]],
    device float* membrane [[buffer(3)]],
    device float* trace [[buffer(4)]],
    device float* threshold [[buffer(5)]],
    device uchar* out [[buffer(6)]],
    constant GateScanParams& p [[buffer(7)]],
    uint batch_index [[thread_position_in_grid]]
) {
    if (batch_index >= p.batch_size || p.dim == 0 || p.dim > GATE_MAX_DIM) {
        return;
    }

    thread uchar raw_active[GATE_MAX_DIM];
    thread float fill_scores[GATE_MAX_DIM];

    const uint state_base = batch_index * p.dim;
    const uint batch_base = batch_index * p.seqlen * p.dim;
    float threshold_value = threshold[batch_index];

    for (uint token_index = 0; token_index < p.seqlen; ++token_index) {
        const uint token_base = batch_base + token_index * p.dim;
        uint raw_count = 0;
        float row_max_fill = 0.0f;
        const float threshold_scale = fmax(fabs(threshold_value), 1.0f);

        for (uint dim_index = 0; dim_index < p.dim; ++dim_index) {
            const uint state_index = state_base + dim_index;
            const float x_value = x[token_base + dim_index];
            const float weight = weights[dim_index];

            const float next_context =
                context[state_index] * p.context_leak + x_value * (1.0f - p.context_leak);
            context[state_index] = next_context;

            const float combined = x_value + p.context_scale * next_context;
            const float next_membrane = membrane[state_index] * p.leak + combined * weight;
            membrane[state_index] = next_membrane;

            const bool active = next_membrane > threshold_value;
            raw_active[dim_index] = active ? 1 : 0;
            raw_count += active ? 1u : 0u;

            const float fill = fabs(x_value) * fabs(weight);
            fill_scores[dim_index] = fill;
            row_max_fill = fmax(row_max_fill, fill);

            const float drive = clamp((next_membrane - threshold_value) / threshold_scale, -4.0f, 4.0f);
            const float soft_fire = 1.0f / (1.0f + exp(-drive));
            const float support = soft_fire * fabs(combined);
            float next_trace = trace[state_index] * p.trace_decay + support;
            if (active) {
                next_trace += support * p.active_bonus;
            }
            trace[state_index] = next_trace;
        }

        const float rate = float(raw_count) / float(p.dim);
        threshold_value = fmax(0.0f, threshold_value + p.threshold_lr * (rate - p.target_rate));
        threshold[batch_index] = threshold_value;

        const uint out_base = token_base;
        if (raw_count > p.max_active) {
            uint kept = 0;
            for (uint dim_index = 0; dim_index < p.dim; ++dim_index) {
                const bool keep = raw_active[dim_index] != 0 && kept < p.max_active;
                out[out_base + dim_index] = keep ? 1 : 0;
                kept += keep ? 1u : 0u;
            }
            continue;
        }

        for (uint dim_index = 0; dim_index < p.dim; ++dim_index) {
            out[out_base + dim_index] = raw_active[dim_index];
        }

        if (raw_count >= p.min_active) {
            continue;
        }

        uint needed = p.min_active - raw_count;
        while (needed > 0) {
            float best_score = -INFINITY;
            int best_index = -1;
            for (uint dim_index = 0; dim_index < p.dim; ++dim_index) {
                if (out[out_base + dim_index] != 0) {
                    continue;
                }
                const float score = fill_scores[dim_index];
                if (score > best_score) {
                    best_score = score;
                    best_index = int(dim_index);
                }
            }
            if (best_index < 0) {
                break;
            }
            out[out_base + uint(best_index)] = 1;
            --needed;
        }
    }
}
)METAL";

id<MTLComputePipelineState> get_gate_scan_pipeline() {
  static id<MTLComputePipelineState> pipeline = nil;
  static std::once_flag once;
  std::call_once(once, [] {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(device != nil, "Failed to acquire a Metal device for gate_scan_mps");

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kGateScanMetalSource];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    TORCH_CHECK(
        library != nil,
        "Failed to compile gate_scan_mps Metal source: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");

    id<MTLFunction> function = [library newFunctionWithName:@"gate_scan_kernel"];
    TORCH_CHECK(function != nil, "Failed to create gate_scan_kernel function");

    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(
        pipeline != nil,
        "Failed to create gate_scan_mps compute pipeline: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
  });
  return pipeline;
}

inline id<MTLBuffer> tensor_buffer(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

inline NSUInteger tensor_offset_bytes(const torch::Tensor& tensor) {
  return static_cast<NSUInteger>(tensor.storage_offset() * tensor.element_size());
}

void check_mps_float_tensor(const torch::Tensor& tensor, const char* name, int64_t dims) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::MPS, name, " must live on MPS");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(tensor.dim() == dims, name, " must have ", dims, " dimensions");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor gate_scan_mps(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor context,
    torch::Tensor membrane,
    torch::Tensor trace,
    torch::Tensor threshold,
    double leak,
    double context_leak,
    double context_scale,
    double threshold_lr,
    double target_rate,
    double trace_decay,
    double active_bonus,
    int64_t min_active,
    int64_t max_active) {
  check_mps_float_tensor(x, "x", 3);
  check_mps_float_tensor(weights, "weights", 1);
  check_mps_float_tensor(context, "context", 2);
  check_mps_float_tensor(membrane, "membrane", 2);
  check_mps_float_tensor(trace, "trace", 2);
  check_mps_float_tensor(threshold, "threshold", 1);

  const auto batch_size = x.size(0);
  const auto seqlen = x.size(1);
  const auto dim = x.size(2);

  TORCH_CHECK(batch_size > 0, "gate_scan_mps requires batch_size > 0");
  TORCH_CHECK(seqlen > 0, "gate_scan_mps requires seqlen > 0");
  TORCH_CHECK(dim > 0, "gate_scan_mps requires dim > 0");
  TORCH_CHECK(dim <= static_cast<int64_t>(kMaxDim), "gate_scan_mps supports dim <= ", kMaxDim);
  TORCH_CHECK(weights.size(0) == dim, "weights dim mismatch");
  TORCH_CHECK(context.size(0) == batch_size && context.size(1) == dim, "context shape mismatch");
  TORCH_CHECK(membrane.sizes() == context.sizes(), "membrane shape mismatch");
  TORCH_CHECK(trace.sizes() == context.sizes(), "trace shape mismatch");
  TORCH_CHECK(threshold.size(0) == batch_size, "threshold shape mismatch");
  TORCH_CHECK(min_active >= 0 && max_active >= 0 && min_active <= max_active, "invalid active bounds");
  TORCH_CHECK(max_active <= dim, "max_active exceeds dim");

  auto output = torch::zeros({batch_size, seqlen, dim}, x.options().dtype(torch::kUInt8));

  GateScanParams params{
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(seqlen),
      static_cast<uint32_t>(dim),
      static_cast<uint32_t>(min_active),
      static_cast<uint32_t>(max_active),
      static_cast<float>(leak),
      static_cast<float>(context_leak),
      static_cast<float>(context_scale),
      static_cast<float>(threshold_lr),
      static_cast<float>(target_rate),
      static_cast<float>(trace_decay),
      static_cast<float>(active_bonus),
  };

  dispatch_queue_t queue = torch::mps::get_dispatch_queue();
  TORCH_CHECK(queue != nullptr, "Failed to acquire the PyTorch MPS dispatch queue");

  dispatch_sync(queue, ^{
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer != nil, "Failed to acquire the current PyTorch MPS command buffer");

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    TORCH_CHECK(encoder != nil, "Failed to create a compute command encoder for gate_scan_mps");

    [encoder setComputePipelineState:get_gate_scan_pipeline()];
    [encoder setBuffer:tensor_buffer(x) offset:tensor_offset_bytes(x) atIndex:0];
    [encoder setBuffer:tensor_buffer(weights) offset:tensor_offset_bytes(weights) atIndex:1];
    [encoder setBuffer:tensor_buffer(context) offset:tensor_offset_bytes(context) atIndex:2];
    [encoder setBuffer:tensor_buffer(membrane) offset:tensor_offset_bytes(membrane) atIndex:3];
    [encoder setBuffer:tensor_buffer(trace) offset:tensor_offset_bytes(trace) atIndex:4];
    [encoder setBuffer:tensor_buffer(threshold) offset:tensor_offset_bytes(threshold) atIndex:5];
    [encoder setBuffer:tensor_buffer(output) offset:tensor_offset_bytes(output) atIndex:6];
    [encoder setBytes:&params length:sizeof(params) atIndex:7];

    const MTLSize threads = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
    const MTLSize group = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:threads threadsPerThreadgroup:group];
    [encoder endEncoding];
  });

  return output;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_scan_mps", &gate_scan_mps, "Hidden dimension gate scan on MPS");
}
