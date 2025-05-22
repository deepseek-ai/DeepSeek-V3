# DeepSeek V3 for MacBook

> This was the initial idea, but now migrating to Zig for better performance and support for any architecture.

This guide provides instructions for running DeepSeek V3 efficiently on MacBook devices with limited resources compared to high-end GPU servers.

## Optimizations Made

The optimized version includes several improvements:

1. **CPU and MPS (Apple Silicon) Support**: Implementation of CPU-compatible kernels and Apple Silicon GPU acceleration.
2. **Memory Efficiency**: Chunked processing and sliding window attention to reduce memory usage.
3. **Quantization**: Optional int8 quantization for CPU to improve inference speed while maintaining reasonable quality.
4. **Reduced Model Size**: Configuration options to load smaller, more efficient model variants.
5. **Dynamic Device Selection**: Automatic selection of the best available device (MPS, CPU).
6. **Progressive Generation**: Ability to see generation progress for longer outputs.

## System Requirements

### Minimum Requirements
- MacBook with Intel CPU (8GB RAM minimum)
- macOS 11 (Big Sur) or newer
- 10GB disk space for model weights

### Recommended
- MacBook with Apple Silicon (M1/M2/M3)
- 16GB RAM or more
- macOS 12 (Monterey) or newer
- 20GB disk space for model weights

## Installation

1. Install dependencies:

```bash
pip install -r inference/requirements_macbook.txt
```

2. Download model weights following instructions in README_WEIGHTS.md

## Usage

The optimized script provides several options to control performance:

```bash
python inference/optimized_generate.py \
  --ckpt-path /path/to/model/weights \
  --config inference/configs/config_macbook.json \
  --interactive \
  --max-new-tokens 200 \
  --temperature 0.2
```

### Additional Options

- `--force-cpu`: Force CPU usage even if GPU is available
- `--no-quantize`: Disable quantization (higher quality but slower)
- `--chunk-size`: Size of processing chunks (default: 512, lower values use less memory)
- `--reduced-model`: Use reduced model parameters (fewer experts/layers for lower resource usage)

## Performance Tips

1. **Context Length**: Keep prompt length short (under 1024 tokens) for better performance.
2. **Batch Size**: Always use batch size of 1 on MacBooks.
3. **Apple Silicon**: M1/M2/M3 MacBooks can use MPS backend for significantly better performance.
4. **Memory Management**: Close other applications when running the model.
5. **Temperature**: Using temperature=0 (greedy decoding) is faster but less creative.

## Troubleshooting

### "Out of Memory" Errors
- Try using `--reduced-model` flag
- Reduce `--chunk-size` to 256 or 128
- Use `--force-cpu` if MPS memory is limited

### Slow Generation
- Ensure you're using the Apple Silicon optimized build of PyTorch
- Check activity monitor to verify GPU utilization
- Try a smaller config file (edit parameters in config_macbook.json)

### Model Loading Errors
- Verify model weights are downloaded correctly
- Ensure safetensors files are in the expected location
- Check torch/transformers versions match requirements
