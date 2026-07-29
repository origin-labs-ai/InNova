# Inference

## Overview

The inference engine runs trained OIL models for text generation with support for KV caching, streaming, and interactive mode.

## Basic Usage

```bash
oil-infer --model model.oil --prompt "Hello, world!"
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-tokens` | 256 | Max tokens to generate |
| `--temperature` | 1.0 | Sampling temperature |
| `--top-p` | 0.95 | Nucleus sampling |
| `--top-k` | 40 | Top-K filtering |
| `--repeat-penalty` | 1.1 | Token repetition penalty |
| `--seed` | -1 | Random seed (-1 = random) |

## Sampling Strategies

### Temperature Scaling
```
T < 1: Sharper distribution, more deterministic
T = 1: Original distribution
T > 1: Flatter distribution, more random
T → 0: Equivalent to greedy (always pick max)
```

### Top-K
Only the K highest probability tokens are considered; rest are masked.

### Top-P (Nucleus)
The smallest set of tokens whose cumulative probability ≥ P is selected.

## Modes

### Streaming
Tokens output in real-time as generated:
```bash
oil-infer --model model.oil --prompt "Hello" --stream
```

### Interactive
Chat-like mode with user input loop:
```bash
oil-infer --model model.oil --interactive
> User: Hello
> Model: Hi! How can I help?
> User: ...
```

## Inference Engine Internals

### KV Cache
The KV cache stores key-value pairs from previous attention computations, avoiding redundant computation.

### Flash Attention
Reduces memory from O(N²) to O(N) by computing attention in tiles without materializing the full N×N matrix.

### Quantized Inference
Quantized weights (OIL4/OIL8) are dequantized on-the-fly during matmul, reducing memory bandwidth by up to 16×.

### GPU Acceleration
Vulkan compute shaders are available for GPU-accelerated inference, offloading GEMM, GELU, and Softmax to the GPU.

## Performance

| Model Size | FP16 | OIL4 (4.0 BPW) | OIL8 (8.0 BPW) |
|-----------|------|-----------------|------------------|
| 7B | 14 GB | 3.5 GB | 7.0 GB |
| 13B | 26 GB | 6.5 GB | 13 GB |
| 70B | 140 GB | 35 GB | 70 GB |

## Related Files

- [Inference Engine](files/engine-inference.cpp.md) — engine implementation
- [Generator](files/generator.h.md) — text generation pipeline
- [Sampler](files/sampler.h.md) — token sampling
- [KV Cache](files/kv_cache.h.md) — key-value caching
- [Optimized Inference](files/inference_opt.h.md) — production config
