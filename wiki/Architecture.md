# Architecture

## Design Philosophy

MYTHOS.cpp is built on four core principles:

1. **Zero Dependencies** — Pure C++20, no Python, PyTorch, Eigen, BLAS, or CUDA libs required
2. **Single Format Truth** — The `.oil` binary format is the single source of truth
3. **Research-First** — Every design decision backed by peer-reviewed papers (see [Research](Research))
4. **Performance-First** — Hand-tuned AVX2 kernels with C++20 fallback

## System Layers

```
┌──────────────────────────────────────────────────────────┐
│                        TOOLS                               │
│  oil-convert  oil-train  oil-infer  oil-finetune        │
│  oil-info     oil-bench                                   │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                       ENGINES                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Inference   │  │    Trainer   │  │   OIL8 Codec │   │
│  │  Engine      │  │  Dense / MoE │  │  Encode/Dec  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                        CORE                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │  Tensor  │ │ Autograd │ │   Math   │ │   MoE    │    │
│  │  (N-D)   │ │   (AD)   │ │(AVX2/C++)│ │ (Sparse) │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │  Memory  │ │  Random  │ │  Types   │ │  Kernel  │    │
│  │  (Pool)  │ │  (RNG)   │ │(Format/  │ │(Quant/   │    │
│  │          │ │          │ │  DType)  │ │Dequant)  │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
└──────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    MODEL LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Transformer  │  │   Backend    │  │  Optimizer   │   │
│  │ (Attention   │  │ (CPU/Vulkan) │  │  (AdamW/SGD) │   │
│  │  + FFN)      │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Tokenizer   │  │   Sampler    │  │  Generator   │   │
│  │  (BPE)       │  │  (top-p/k)   │  │  (Pipeline)  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Key Data Flow

### Training Flow
```
Data → Tokenizer → DataLoader → Model Forward → Loss → Autograd Backward → Optimizer → Checkpoint
```

### Inference Flow
```
Prompt → Tokenizer → KV Cache Init → [ Decode Loop ] → Sampler → Detokenizer → Output
                                    ↓
                            Model Forward (cached)
```

## Component Dependencies

```
oil_types        (foundation types, no deps)
oil_memory        → oil_types
oil_random        → oil_types
oil_tensor        → oil_types, oil_memory
oil_autograd      → oil_tensor
oil_math          → oil_tensor (AVX2 optional)
oil_kernel        → oil_tensor, oil_types
oil_model         → oil_tensor, oil_math, oil_types
oil_transformer   → oil_model, oil_math, oil_tensor
oil_backend       → oil_types
oil_tokenizer     → oil_types
oil_sampler       → oil_tensor, oil_random
oil_generator     → oil_model, oil_tokenizer, oil_sampler, oil_kv_cache
oil_trainer       → oil_model, oil_optimizer, oil_dataloader
oil_format        → oil_types, oil_tensor
oil_oil8          → oil_tensor, oil_types
oil_moe           → oil_model, oil_tensor
oil_gpu           → oil_tensor, oil_backend
oil_asi           → oil_model, oil_moe
oil_inference_opt → oil_model, oil_flash_attention
```

## Quantization Pipeline

```
FP32 Model → Calibrate → Analyze Sensitivities → Assign Formats → Quantize → OIL File
                                                      │
                                              ┌───────┴────────┐
                                              ▼                ▼
                                          OIL4 (4-bit)    OIL8 (8-bit)
                                          1.50 BPW         0.85 BPW
```

## Build Time Dependencies

| Dependency | Required? | Purpose |
|-----------|-----------|---------|
| CMake ≥ 3.24 | Yes | Build system |
| C++20 compiler | Yes | Language standard |
| Ninja | No | Faster builds |
| Vulkan SDK | No | GPU acceleration (primary backend) |
