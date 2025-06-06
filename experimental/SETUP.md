# DeepZig V3 Implementation - Setup Guide

This guide will help you set up the development environment and understand the project structure.

## Prerequisites

### 1. Install Zig 0.15.0-dev

Download the latest development build from [ziglang.org/download](https://ziglang.org/download/):

```bash
# macOS (using Homebrew)
brew install zig --HEAD

# Linux (manual installation)
wget https://ziglang.org/builds/zig-linux-x86_64-0.15.0-dev.xxx.tar.xz
tar -xf zig-linux-x86_64-0.15.0-dev.xxx.tar.xz
export PATH=$PATH:/path/to/zig

# Verify installation
zig version
# Should show: 0.15.0-dev.xxx
```

### 2. Platform-Specific Setup

#### macOS (for Metal backend)
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal
```

#### Linux (for CUDA backend, optional)
```bash
# Install CUDA Toolkit (optional)
# Follow: https://developer.nvidia.com/cuda-downloads

# For Ubuntu/Debian:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

## Project Overview

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Layer     │    │   Core Engine    │    │   Backends      │
│                 │    │                  │    │                 │
│ ├─ HTTP API     │◄──►│ ├─ Transformer   │◄──►│ ├─ CPU (SIMD)   │
│ ├─ WebSocket    │    │ ├─ Attention     │    │ ├─ Metal (macOS)│
│ ├─ Rate Limit   │    │ ├─ MoE Routing   │    │ ├─ CUDA (Linux) │
│ └─ Auth         │    │ └─ Tokenizer     │    │ └─ WebGPU       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

#### Core Module (`src/core/`)
- **Tensor Operations**: SIMD-optimized tensor math with AVX2/NEON support
- **Model Architecture**: DeepSeek V3 implementation with MLA and MoE
- **Memory Management**: Arena allocators and memory pools
- **Backend Abstraction**: Unified interface for CPU/GPU computation

#### Web Layer (`src/web/`)
- **HTTP Server**: Built on `std.http.Server` (Zig 0.15.0 compatible)
- **OpenAI API**: Compatible `/v1/chat/completions` endpoint
- **Middleware**: CORS, authentication, rate limiting
- **WebSocket**: Streaming inference support (planned)

#### Backends (`src/backends/`)
- **CPU**: Multi-threaded with SIMD optimizations
- **Metal**: Apple Silicon GPU acceleration (macOS)
- **CUDA**: NVIDIA GPU support with Tensor Cores (Linux/Windows)

## Development Workflow

### 1. Initial Setup

```bash
# Clone the repository
cd experimental/

# Build the project
zig build

# Run tests to verify setup
zig build test

# Run benchmarks
zig build bench
```

### 2. Development Commands

```bash
# Format code
zig fmt src/

# Run tests with watch mode (in development)
zig build test

# Build optimized release
zig build -Doptimize=ReleaseFast

# Cross-compile for different targets
zig build -Dtarget=aarch64-macos    # Apple Silicon
zig build -Dtarget=x86_64-linux     # Linux x64
zig build -Dtarget=wasm32-freestanding # WebAssembly
```

### 3. Running the Server

```bash
# Default configuration (CPU backend, port 8080)
zig build run

# Custom configuration
zig build run -- --port 3000 --backend metal

# With model path (when implemented)
zig build run -- --model ./models/deepseek-v3.bin --backend cuda
```

### 4. Testing the API

```bash
# Health check
curl http://localhost:8080/health

# Model information
curl http://localhost:8080/v1/models

# Chat completion (placeholder response)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Implementation Status

### ✅ Ready for Development
- [x] Build system and project structure
- [x] Core tensor operations with SIMD
- [x] HTTP server with basic routing
- [x] OpenAI API compatibility layer
- [x] Memory management utilities
- [x] Benchmark framework

### 🚧 Needs Implementation
- [ ] **DeepSeek V3 Model**: Transformer architecture
- [ ] **Attention Mechanism**: Multi-Head Latent Attention (MLA)
- [ ] **MoE Implementation**: Expert routing and selection
- [ ] **Tokenizer**: BPE tokenization (currently placeholder)
- [ ] **Model Loading**: Weight file parsing and loading
- [ ] **GPU Backends**: Metal and CUDA kernel implementations

### 📋 Future Enhancements
- [ ] Model quantization (INT8, FP16)
- [ ] Flash Attention optimization
- [ ] WebSocket streaming
- [ ] Distributed inference
- [ ] Model sharding

## Code Style and Conventions

### Zig Best Practices
- Use `snake_case` for functions and variables
- Use `PascalCase` for types and structs
- Prefer explicit error handling with `!` and `catch`
- Use arena allocators for request-scoped memory
- Leverage comptime for zero-cost abstractions

### Error Handling
```zig
// Preferred: explicit error handling
const result = someFunction() catch |err| switch (err) {
    error.OutOfMemory => return err,
    error.InvalidInput => {
        std.log.err("Invalid input provided");
        return err;
    },
    else => unreachable,
};

// Use defer for cleanup
var tensor = try Tensor.init(allocator, shape, .f32);
defer tensor.deinit();
```

### Memory Management
```zig
// Use arena allocators for request scope
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
const request_allocator = arena.allocator();

// Use memory pools for tensors
var tensor_pool = TensorPool.init(allocator);
defer tensor_pool.deinit();
```

## Performance Considerations

### SIMD Optimization
- Use `@Vector` types for SIMD operations
- Align data to cache line boundaries (64 bytes)
- Prefer blocked algorithms for better cache locality

### Backend Selection
- **CPU**: Best for smaller models, development
- **Metal**: Optimal for Apple Silicon (M1/M2/M3)
- **CUDA**: Best for NVIDIA GPUs with Tensor Cores

### Memory Layout
- Use structure-of-arrays (SoA) for better vectorization
- Minimize memory allocations in hot paths
- Leverage unified memory on Apple Silicon

## Debugging and Profiling

### Debug Build
```bash
# Build with debug symbols
zig build -Doptimize=Debug

# Run with verbose logging
RUST_LOG=debug zig build run
```

### Performance Profiling
```bash
# Run benchmarks
zig build bench

# Profile with system tools
# macOS: Instruments.app
# Linux: perf, valgrind
# Windows: Visual Studio Diagnostics
```

## Next Steps

1. **Choose an area to implement**:
   - Core ML components (transformer, attention, MoE)
   - Backend optimizations (Metal shaders, CUDA kernels)
   - Web features (streaming, authentication)

2. **Read the code**:
   - Start with `src/core/root.zig` for module structure
   - Check `src/main.zig` for the server entry point
   - Look at `bench/main.zig` for performance testing

3. **Run and experiment**:
   - Build and run the server
   - Try the API endpoints
   - Run benchmarks to understand performance
   - Read the TODOs in the code for implementation ideas

4. **Contribute**:
   - Pick a TODO item
   - Implement and test
   - Submit improvements

## Resources

- [Zig Language Reference](https://ziglang.org/documentation/master/)
- [DeepSeek V3 Paper](https://arxiv.org/abs/2412.19437)
- [Zig SIMD Guide](https://ziglang.org/documentation/master/#Vectors)
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

Ready to build the future of high-performance LLM inference! 🚀 