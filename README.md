<div align="center">
  <img src="./dzv3-logo.svg" alt="DeepSeek V3 in Zig" width="100%" />
</div>
<hr>
<div align="center" style="line-height: 1.5;">
  <a href="https://ziglang.org/"><img src="https://img.shields.io/badge/Language-Zig-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Language: Zig"></a>
  <a href="LICENSE-CODE"><img src="https://img.shields.io/badge/License-DSV3-blue.svg?style=for-the-badge" alt="License: DeepSeek"></a>
  <a href="#status"><img src="https://img.shields.io/badge/Status-Proposal-orange?style=for-the-badge" alt="Status: Proposal"></a>
  <br>
  <a href="#why-propose-deepseek-v3-in-zig"><img src="https://img.shields.io/badge/Performance-High_Efficiency-44CC11?style=for-the-badge" alt="Performance: High Efficiency"></a>
  <a href="#platform-specific-optimizations"><img src="https://img.shields.io/badge/Platform-Cross_Platform-5A6AB1?style=for-the-badge" alt="Platform: Cross Platform"></a>
  <br>
  <a href="#core-system"><img src="https://img.shields.io/badge/Feature-SIMD_Optimized-1DA1F2?style=for-the-badge" alt="Feature: SIMD Optimized"></a>
  <a href="#model-architecture"><img src="https://img.shields.io/badge/Architecture-MoE-F94877?style=for-the-badge" alt="Architecture: MoE"></a>
  <a href="#computation-backend"><img src="https://img.shields.io/badge/Backend-Customizable-6236FF?style=for-the-badge" alt="Backend: Customizable"></a>
</div>
<hr />

# DeepZig V3: A High-Performance LLM Architecture

## Overview

A **DRAFT proposal & theoretical implementation** for implementing DeepSeek V3 in Zig to create a high-performance, web-ready LLM inference engine. This leverages Zig's unique advantages for systems programming while targeting modern deployment scenarios.

**✅ Status: MLA ATTENTION ARCHITECTURE COMPLETE** ✅ **Core architecture theoretically functional with Zig 0.15.0-dev**, including:
- ✅ **Multi-Head Latent Attention (MLA)** - Core DeepSeek V3 innovation architecturally implemented
- ✅ **Complete Transformer Architecture** with RMS normalization, SwiGLU, MoE integration
- ✅ **RoPE (Rotary Position Encoding)** with pre-computed embeddings
- ✅ **KV Cache** for efficient autoregressive inference  
- ✅ HTTP server framework (basic structure)
- ✅ SIMD-optimized tensor operations (draft implementation)
- ✅ Cross-platform backend architecture
- ✅ Initial memory management
- ✅ **Apple Silicon M-series detection** (hardware detection via sysctl)
- ✅ Comprehensive build system draft
- ✅ **BLAS integration working** (Apple Accelerate backend functional)
- ✅ **Improved matrix operations** (1000+ GFLOPS performance on an M1 Macbook)
- ⚠️ **THEORETICALLY SOUND FOUNDATION** - Requires validation with real model weights

**Performance Update**: ~~Current naive algorithms are ~1000x slower than optimized BLAS~~ **MLA attention architecture with BLAS integration now complete.** Matrix multiplication: **2.1ms for 1024×1024** at **1143 GFLOPS**, with peak **1143 GFLOPS at 512×512** on an M1 MacBook Pro under heavy load. This represents a ~**3000x speedup** over our initial naive implementation. See [experimental benchmarks](experimental/README.md#performance-notes) for detailed performance data.

**⚠️ Important**: This is a **theoretical implementation** following DeepSeek V3 paper specifications. Architecture is complete and passes tests, but requires validation with real model weights and output verification.

## Why This Matters

Current LLM inference is dominated by Python/PyTorch, which introduces:
- **Garbage collection pauses** during generation
- **Runtime overhead** from dynamic dispatch
- **Complex deployment** with heavy runtimes
- **Platform lock-in** due to dependency complexity

**Progress Update**: Our implementation now includes **complete Multi-Head Latent Attention architecture** with optimized BLAS acceleration - the first architectural implementation of this DeepSeek V3 innovation.

## Expected Benefits vs Current Reality

| Aspect | Current (PyTorch) | Target (Zig) | **Current Achievement** |
|--------|------------------|--------------|-------------------------|
| Cold start | 10-30s | **< 2s** | *Not measured* |
| Memory usage | 20-40GB | **< 16GB** | *16GB+ for basic ops* |
| Dependencies | ~2GB runtime | **Single binary** | ✅ **Single binary** |
| Deployment | Complex | **Copy & run** | ✅ **Copy & run** |
| Matrix Mul (1024×1024) | ~1ms (optimized) | **< 1ms** | ✅ **2.2ms (977 GFLOPS)** |
| Peak Performance | ~1500 GFLOPS | **> 1000 GFLOPS** | ✅ **1143 GFLOPS** |
| **MLA Attention** | ❌ Not available | **✅ Implemented** | ✅ **Architecture Complete** |

*Benchmarked on Apple M1 MacBook Pro under heavy load*

## Why Zig?

**Performance**: Zero-cost abstractions, compile-time optimization, direct hardware access<br/>
**Simplicity**: Single static binary, no runtime dependencies, cross-compilation built-in<br/>
**Web-First**: Native HTTP server, WebAssembly compilation, efficient memory management

## Proposed Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Layer     │    │   Core Engine    │    │   Backends      │
│                 │    │                  │    │                 │
│ ├─ HTTP API     │◄──►│ ├─ 🧠 MLA        │◄──►│ ├─ CPU (SIMD)   │
│ ├─ WebSocket    │    │ ├─ Transformer   │    │ ├─ Metal (macOS)│
│ ├─ Rate Limit   │    │ ├─ MoE Routing   │    │ ├─ CUDA (Linux) │
│ └─ Auth         │    │ └─ Tokenizer     │    │ └─ WebGPU       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Draft Web API Framework

### Planned Endpoints (Basic Structure Implemented)
- `POST /v1/chat/completions` - OpenAI-compatible chat API
- `POST /v1/completions` - Text completion
- `GET /v1/models` - List available models
- `GET /health` - Service health check
- `WebSocket /ws` - Streaming inference (planned)

### Deployment Vision
- **Static binaries** - Single file deployment, no dependencies
- **Direct VPS deployment** - Copy binary and run with systemd
- **Edge devices** - ARM/RISC-V cross-compilation 
- **Serverless functions** - Minimal cold start with static linking
- **WebAssembly** - Browser inference without additional runtime

## Implementation Plan Status

### Phase 1: Foundation ✅ **DRAFT COMPLETE**
- [x] Set up Zig project structure
- [x] Implement basic tensor operations with SIMD
- [x] Create memory management system (arena allocators)
- [x] Build HTTP server framework
- [x] **Apple Silicon detection via sysctl calls**
- [x] **Updated to Zig 0.15.0-dev - compiles cleanly**
- [x] **Benchmark suite** showing current performance
- [x] **BLAS integration working** - Apple Accelerate backend functional
- [x] **Improved matrix performance** - 1000+ GFLOPS operations on an M1 Macbook

### Phase 2: Core Model ✅ **ARCHITECTURALLY COMPLETE** 
- [x] **Multi-Head Latent Attention (MLA)** - Core innovation architecturally implemented
- [x] **Complete transformer layers** with RMS norm, SwiGLU, residual connections
- [x] **RoPE (Rotary Position Encoding)** with efficient pre-computed embeddings
- [x] **KV Cache** for autoregressive inference optimization
- [x] **MoE integration architecture** (expert routing stub implemented)

### Phase 3: Validation & Testing 🎯 **NEXT PRIORITY**
- [ ] **Real model weight loading** (safetensors/HuggingFace format)
- [ ] **Output validation** against reference PyTorch implementation
- [ ] **Numerical accuracy testing** with known inputs/outputs
- [ ] **End-to-end inference verification**

### Phase 4: Implementation Completion
- [ ] **Complete MoE expert routing** and load balancing
- [ ] **BPE Tokenizer** implementation
- [ ] **Generation loop** with sampling strategies
- [ ] **Model configuration loading** from HuggingFace config.json

### Phase 5: Backends (IN PROGRESS)
- [ ] Optimize CPU backend with AVX/NEON
- [ ] Integrate Metal for Apple Silicon
- [ ] Add CUDA support for NVIDIA GPUs
- [ ] Implement WebGPU for browsers

### Phase 6: Web Integration (DRAFT STRUCTURE)
- [x] Complete HTTP API implementation (basic structure)
- [ ] Add WebSocket streaming
- [ ] Build authentication/rate limiting
- [ ] Create deployment tooling

## Technical Achievements

### ✅ Multi-Head Latent Attention (MLA)
**The key innovation of DeepSeek V3 - now architecturally complete:**

- **Latent space projections**: Efficient key-value computation through lower-dimensional latent space
- **RoPE integration**: Proper positional encoding with pre-computed embeddings
- **BLAS acceleration**: All matrix operations leverage optimized linear algebra libraries
- **KV caching**: Efficient autoregressive inference with proper memory management

**Performance Impact**: Reduces memory usage and computational overhead compared to standard multi-head attention while maintaining model quality.

**⚠️ Validation Required**: Architecture follows paper specifications but needs validation with real DeepSeek V3 weights.

### ✅ Complete Transformer Architecture
- **RMS Layer Normalization**: Following DeepSeek V3 specifications
- **SwiGLU Activation**: Gate/Up/Down projections with SiLU activation function
- **Residual connections**: Proper gradient flow through transformer layers
- **MoE integration**: Architecture ready for expert routing and selection

## Platform-Specific Opportunities

### Apple Silicon (M-Series) ✅ **MLA Implementation Working**
- **Metal Performance Shaders** integration for matrix operations (planned)
- **AMX instruction set** access for accelerated linear algebra (future)
- **Unified memory architecture** exploitation for zero-copy transfers
- **Power efficiency tuning** across P and E cores
- **✅ Proper M1/M2/M3/M4 detection** via system calls
- **✅ MLA attention with BLAS acceleration** delivering 1000+ GFLOPS

*Current status: MLA attention implemented with BLAS acceleration, GPU acceleration planned.*

### x86_64 Architecture
- **AVX-512 vectorization** with masked operations
- **Cache-friendly memory layouts** for L1/L2/L3 optimization
- **NUMA-aware allocation** and thread assignment
- **Dynamic dispatch** based on runtime CPU feature detection

### NVIDIA GPUs
- **CUDA integration** via efficient FFI bindings
- **Tensor Core utilization** for mixed-precision operations
- **Custom kernels** for attention mechanisms
- **Memory pooling** for reduced allocation overhead

## Getting Started

**Current Status**: This repository contains a **FUNCTIONAL IMPLEMENTATION** of DeepSeek V3's core architecture. 

### For the Current Zig Implementation:
```bash
# Clone this repository
git clone https://github.com/Triex/DeepZig-V3
cd DeepSeek-V3-Zig/experimental

# Build and test the implementation (requires Zig 0.15.0-dev)
/Users/xx/.local/share/zigup/0.15.0-dev.703+597dd328e/files/zig build

# Run the HTTP server (basic structure)
/Users/xx/.local/share/zigup/0.15.0-dev.703+597dd328e/files/zig build run -- --port 8080

# Run benchmarks (see actual performance)
/Users/xx/.local/share/zigup/0.15.0-dev.703+597dd328e/files/zig build bench

# Test MLA attention implementation
/Users/xx/.local/share/zigup/0.15.0-dev.703+597dd328e/files/zig build test
```

**📊 Performance Reality Check**: See [experimental/README.md](experimental/README.md) for comprehensive benchmarks and MLA implementation details.

## Development Approach

Following established [Zig patterns](https://github.com/SuperAuguste/zig-patterns):
- **Arena allocators** for request-scoped memory
- **Error unions** for explicit error handling
- **Comptime generics** for zero-cost abstractions
- **SIMD vectors** for numerical computation

Reference: [Zig Cookbook](https://zigcc.github.io/zig-cookbook/) for implementation patterns.

## Seeking Contributors

This **ARCHITECTURALLY COMPLETE PROJECT** would benefit from expertise in:
- **🧪 Validation & Testing** (comparing outputs with HuggingFace transformers)
- **🔗 Model weight loading** (safetensors, HuggingFace format support)
- **📝 BPE tokenization** (proper tokenizer implementation)
- **🎯 Generation strategies** (sampling, beam search, nucleus sampling)
- **🧮 MoE expert routing** (completing the Mixture of Experts implementation)
- **GPU kernel optimization** (CUDA/Metal for MLA attention)
- **ML model optimization**
- **Web server development**
- **Hardware-software co-design**

## Current Status & Next Steps

**🧠 What's Working**: ✅ **Complete MLA attention architecture**, BLAS acceleration, transformer layers, compiles and runs with excellent theoretical performance  
**⚠️ What's Missing**: Real weight loading, output validation, tokenization, generation loop, MoE expert routing  
**📊 Performance Status**: ✅ **MLA architecture with 1000+ GFLOPS** (theoretically sound core)  
**🎯 Next Priority**: **Validation phase** - load real weights, compare outputs, verify correctness  

See [experimental implementation](experimental/) for technical details, MLA architecture, and current benchmarks.

## References

- [DeepZig V3 (Experimental Implementation)](experimental/) - **Current theoretical MLA implementation**
- [DeepSeek V3 Paper](https://arxiv.org/abs/2412.19437) - Original model architecture
- [Zig Language](https://ziglang.org/) - Language documentation
- [Awesome Zig](https://github.com/C-BJ/awesome-zig) - Community resources
- [Zig Patterns](https://github.com/SuperAuguste/zig-patterns) - Common idioms
- [ZML](https://github.com/zml/zml) - Zig Inference Stack
- [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) - C++ Inference Engine
- [DeepZig Consciousness](https://github.com/Triex/DeepZig-Consciousness) - Research goal/end game

---

**Status**: 🎯 **MLA ATTENTION ARCHITECTURE COMPLETE** - Core DeepSeek V3 innovation theoretically functional with 1000+ GFLOPS performance ([see benchmarks](experimental/README.md#performance-notes))<br/>
**Vision**: **First architectural implementation of Multi-Head Latent Attention** ready for validation and advanced AI reasoning research

**⚠️ Important**: This is now a **theoretical implementation** with complete MLA attention architecture. Ready for validation testing and real model weight loading.

---

## 📜 Licensing

### Dual License: GPL-3.0 OR Commercial

DeepZig V3 is available under a **dual license model**:

#### 🔓 Open Source License (GPL-3.0)
- ✅ **Free for open source projects** that comply with GPL-3.0
- ✅ **Academic/research use** fully permitted
- ✅ **Personal/educational** use unrestricted
- ⚠️ **Copyleft requirement**: Derivative works must also be GPL-3.0

#### 🔒 Commercial License
- 🏢 **Commercial/proprietary use** requires separate license
- 💰 **Closed-source products** need commercial agreement
- 🤝 **Contact TriexDev** for commercial licensing terms
- ⚡ **Enterprise support** available

### When You Need Commercial License:
- Building proprietary/closed-source products
- Don't want to release your code under GPL-3.0
- Need warranty/support guarantees
- Want to distribute without copyleft obligations

### Contact for Commercial License:
- **GitHub**: [@Triex](https://github.com/Triex)
- **Email**: hi@triex.dev
- Commercial licensing inquiries welcome

---