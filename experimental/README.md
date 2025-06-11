# DeepZig V3 Implementation 🚀

A high-performance implementation of DeepSeek V3 in [Zig](https://ziglang.org/) for blazingly fast inference.

> **⚠️ Status: Experimental Foundation** 
> 
> This project provides an **experimental foundation** for DeepZig V3 with working draft implementation:
> - ✅ **HTTP server** with OpenAI-compatible API
> - ✅ **BLAS-accelerated tensor operations** (Apple Accelerate working)
> - ✅ **Cross-platform build system** (Zig 0.15.0-dev)
> - ✅ **Memory management** and backend architecture
> - ✅ **Apple Silicon detection and optimization**
> - ✅ **Functional matrix operations** (significant performance improvement)
> 
> **Recent Progress**: Matrix operations now use BLAS acceleration<br/>
> **Performance Status**: 1160+ GFLOPS with Apple Accelerate backend working (measured on Apple M1)<br/>
> 
> See [Performance Results](#performance-notes) for detailed benchmarks.

## Overview

This experimental implementation aims to leverage Zig's unique advantages for systems programming to create a high-performance LLM inference engine:

- **Zero-cost abstractions** with compile-time optimization
- **Direct hardware access** for SIMD and platform-specific optimizations  
- **Manual memory management** without garbage collection pauses
- **Single binary deployment** with no runtime dependencies
- **Cross-platform compilation** for multiple architectures

**🚀 BLAS Acceleration Achieved!** We've successfully integrated Apple Accelerate backend delivering **1000+ GFLOPS** performance - a **3000x speedup** over the initial naive implementation.

**🔗 Related**: See the [main project README](../README.md) for architecture overview and vision.

## Project Structure

```
experimental/
├── build.zig              # Build system configuration
├── build.zig.zon          # Package dependencies  
├── src/
│   ├── main.zig           # HTTP server entry point
│   ├── core/              # Core ML components
│   │   ├── root.zig       # Module exports
│   │   ├── tensor.zig     # SIMD-optimized tensors
│   │   ├── model.zig      # DeepSeek V3 model
│   │   ├── attention.zig  # MLA attention mechanism
│   │   ├── moe.zig        # Mixture of Experts
│   │   ├── tokenizer.zig  # Text tokenization
│   │   ├── backend.zig    # Backend abstraction
│   │   ├── memory.zig     # Memory management
│   │   └── math/          # Math utilities
│   │       ├── root.zig   # Math module exports
│   │       ├── simd.zig   # SIMD operations
│   │       ├── activation.zig  # Activation functions
│   │       └── rms_norm.zig    # RMS normalization
│   ├── web/               # HTTP API layer
│   │   ├── root.zig       # Web module exports
│   │   ├── server.zig     # HTTP server (std.http)
│   │   ├── handlers.zig   # Request handlers
│   │   ├── middleware.zig # CORS, auth, rate limiting
│   │   ├── websocket.zig  # WebSocket support
│   │   ├── openai.zig     # OpenAI API compatibility
│   │   ├── request.zig    # Request wrapper
│   │   └── response.zig   # Response wrapper
│   ├── backends/          # Compute backends
│   │   ├── cpu/           # CPU with SIMD
│   │   ├── metal/         # Apple Silicon
│   │   └── cuda/          # NVIDIA GPUs
│   └── wasm/
│       └── main.zig       # WebAssembly entry point
├── bench/
│   └── main.zig           # Performance benchmarks
└── README.md               # This file
```

## Requirements

- **Zig 0.15.0-dev**
- Platform-specific requirements:
  - **macOS**: Xcode Command Line Tools (for Metal backend)
  - **Linux**: CUDA Toolkit (for CUDA backend, optional)
  - **Windows**: CUDA Toolkit (for CUDA backend, optional)

## Quick Start

### Building

```bash
# Clone and navigate to experimental directory
cd experimental/

# Build the project
zig build

# Run the server
zig build run

# Run tests
zig build test

# Run benchmarks
zig build bench

# Build WebAssembly
zig build wasm
```

### Running the Server

```bash
# Start server on default port (8080)
./zig-out/bin/deepseek-v3-zig

# Custom configuration
./zig-out/bin/deepseek-v3-zig --port 3000 --backend metal --model ./path/to/model
```

### API Usage

The server exposes OpenAI-compatible endpoints:

```bash
# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Health check
curl http://localhost:8080/health

# Model info
curl http://localhost:8080/v1/models
```

## Performance Features

### SIMD Optimizations

- **x86_64**: AVX2/AVX-512 vectorization for matrix operations
- **ARM64**: NEON SIMD for Apple Silicon optimization
- **Auto-vectorization**: Compiler-optimized loops with `@Vector` types

### Backend Support

| Backend | Status | Features |
|---------|--------|----------|
| **CPU** | ✅ Implemented | Multi-threaded, SIMD, cache-optimized |
| **Metal** | 🚧 In Progress | Apple Silicon GPU, unified memory |
| **CUDA** | 🚧 Planned | NVIDIA GPU, Tensor Cores |
| **WebGPU** | 📋 Future | Browser GPU acceleration |

### Memory Management

- **Arena allocators** for request-scoped memory
- **Memory pools** for tensor allocations
- **Zero-copy operations** where possible
- **Cache-friendly** data layouts

## Development Status

### ✅ Drafted
- [x] Project structure and build system
- [x] Core tensor operations with SIMD
- [x] HTTP server with OpenAI API compatibility
- [x] CPU backend with optimizations
- [x] Memory management utilities
- [x] Benchmark suite

### 🚧 In Progress
- [ ] DeepSeek V3 model architecture
- [ ] Multi-Head Latent Attention (MLA)
- [ ] Mixture of Experts (MoE) implementation
- [ ] Metal backend for Apple Silicon
- [ ] Model loading and weight management

### 📋 Planned
- [ ] CUDA backend for NVIDIA GPUs
- [ ] WebSocket streaming
- [ ] Model quantization (INT8, FP16)
- [ ] Flash Attention optimization
- [ ] Distributed inference
- [ ] Advanced sampling strategies

## Architecture Decisions

### Why Zig?

1. **Performance**: Zero-cost abstractions without runtime overhead
2. **Memory Safety**: Compile-time memory management without GC
3. **Simplicity**: Single binary deployment, cross-compilation
4. **Control**: Direct hardware access for optimization

### Design Principles

- **Modularity**: Clean separation between core, web, and backend layers
- **Performance**: SIMD-first design with cache-friendly algorithms  
- **Compatibility**: OpenAI API compatibility for easy adoption
- **Extensibility**: Plugin architecture for new backends

## Contributing

This is an experimental project! Contributions are welcome:

1. **Core ML**: Implement transformer layers, attention mechanisms
2. **Backends**: Optimize CUDA/Metal compute kernels
3. **Performance**: Profile and optimize bottlenecks
4. **Testing**: Add comprehensive test coverage
5. **Documentation**: Improve setup and usage guides

### Development Setup

```bash
# Install Zig 0.15.0-dev
# https://ziglang.org/download/

# Clone repository
git clone [repository-url]
cd experimental/

# Run tests during development
zig build test --watch

# Format code
zig fmt src/
```

## Benchmarks

Run benchmarks to measure performance:

```bash
zig build bench
```

**Hardware Context**: Benchmarks run on Apple M1 MacBook Pro (MacBookPro17,1) with 16GB unified memory, Zig 0.15.0-dev.703+597dd328e, debug build.

Example output:
```
🚀 DeepZig V3 Performance Benchmarks
==========================================

Backend: CPU (BLAS accelerated)
Architecture: aarch64  
Thread count: 8
Hardware: Apple M1 MacBook Pro, 16GB unified memory

Operation                      | Iterations |  Avg Time | Operations/s | Memory
-------------------------------|------------|-----------|--------------|-------
Tensor Creation (1024x1024)    |   1000 iter |     2.03 ms |        493 ops/s |   4.0 MB
Tensor Addition (SIMD)         |    100 iter |     1.49 ms | 2806962690 ops/s |  48.0 MB  
Matrix Multiplication (BLAS)   |     10 iter |     2.1 ms  |      1164 GFLOPS |  12.0 MB
SwiGLU Activation              |   1000 iter |     4.44 ms |  236002478 ops/s |   12.0 MB
RMS Normalization (SIMD)       |   1000 iter |     0.00 ms |    1077586 ops/s |    0.0 MB
Memory Bandwidth               |    100 iter |     4.92 ms |         13 ops/s |  128.0 MB
```

## Known Issues

- **Model Loading**: Currently creates dummy models - real weight loading not implemented
- **Tokenizer**: Placeholder implementation - needs proper BPE tokenizer
- **WebSocket**: Basic structure only - streaming not implemented
- **Metal/CUDA**: Backend stubs only - GPU kernels not implemented

## License

This experimental implementation follows the same license as the original DeepSeek V3 project.

## Resources

- [Original DeepSeek V3 Paper](https://arxiv.org/abs/2412.19437)
- [Zig Language Documentation](https://ziglang.org/documentation/master/)
- [Zig Performance Guide](https://github.com/ziglang/zig/wiki/Performance)
- [SIMD in Zig](https://ziglang.org/documentation/master/#Vectors)

## Is This Ready for Production? 

**No** - this is a research/development foundation. But it's **theoretical and compiles**:

- **What works now**: ✅ Compiles and runs with Zig 0.15.0-dev, HTTP server, tensor operations, SIMD math, benchmarks execute successfully
- **What's missing**: Optimized matrix operations, actual DeepSeek V3 model implementation
- **Timeline**: Foundation is **compiling**, model implementation is the next major milestone

## Comparison to Other Projects

| Project | Language | Status | Focus |
|---------|----------|--------|-------|
| **This** | Zig | Foundation + API | Web-first inference |
| llama.cpp | C++ | Production | CLI/library |
| Candle | Rust | Production | ML framework |
| ZML | Zig | Research | Low-level ML ops |

**Unique advantages**: Built-in web server, Zig's zero-cost abstractions, single binary deployment.

---

**⚡ Built with Zig for blazing fast LLM inference!** 

## Performance Notes

**Current Status**: ✅ **BLAS integration working** - Apple Accelerate backend now functional in draft implementation.

**Performance Results** (Apple M1, Accelerate backend):
- **Matrix 256×256**: 0.1ms/iter, **561 GFLOPS** (21.6% efficiency)
- **Matrix 512×512**: 0.2ms/iter, **1129 GFLOPS** (43.4% efficiency)  
- **Matrix 1024×1024**: 2.1ms/iter, **1004 GFLOPS** (38.6% efficiency)
- **Matrix 2048×2048**: 21.5ms/iter, **799 GFLOPS** (30.7% efficiency)

**Performance Improvement**: From **6418ms naive** → **2.1ms BLAS** = significant speedup for matrix operations

**System Status**:
- ✅ **BLAS Backend**: Apple Accelerate integration working
- ✅ **Efficiency**: 20-44% of theoretical maximum (good for draft implementation)
- ✅ **Memory Bandwidth**: 23.5 GB/s copying, basic optimization
- ✅ **Hardware Detection**: M-series Apple Silicon detection functional

**Next Steps**: Focus on transformer architecture, attention mechanisms, and model-specific optimizations for the draft DeepSeek V3 implementation. 