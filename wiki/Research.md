# Research

## Overview

InNova is built on peer-reviewed research. All research documents are in `.research/`.

## Research Papers

| # | Paper | Topic | Key Result |
|---|-------|-------|------------|
| 01 | [BitNet b1.58 Analysis](file:///c:/Users/thaku/Downloads/InNova/.research/01-bitnet-b1.58-analysis.md) | 1.58-bit quantization | Extreme low-bit LLMs viable |
| 02 | [BitNetCPP Lossless Analysis](file:///c:/Users/thaku/Downloads/InNova/.research/02-bitnetcpp-lossless-analysis.md) | Lossless compression | Additional gains beyond quantization |
| 04 | [VQ-VAE Residual Coding](file:///c:/Users/thaku/Downloads/InNova/.research/04-vqvae-residual-coding.md) | Residual VQ | Progressive quality improvement |
| 05 | [Rate-Distortion Bounds](file:///c:/Users/thaku/Downloads/InNova/.research/05-rate-distortion-bounds.md) | Compression theory | Theoretical quality guarantees |
| 06 | [STE Training in Format](file:///c:/Users/thaku/Downloads/InNova/.research/06-ste-training-in-format.md) | STE quantization | Train quantized-aware models |
| 07 | [Dynamic Routing GAMPs](file:///c:/Users/thaku/Downloads/InNova/.research/07-dynamic-routing-gamps.md) | MoE routing | Adaptive expert selection |
| 08 | [Final Verdict](file:///c:/Users/thaku/Downloads/InNova/.research/08-final-verdict.md) | Summary | Cross-method analysis |
| 09 | [OIL8 256 Centroids](file:///c:/Users/thaku/Downloads/InNova/.research/09-oil8-256-centroids.md) | OIL8 quantization | 256-centroid analysis |

## Key Results

| Method | BPW | WikiText-2 PPL (vs FP16) |
|--------|-----|--------------------------|
| FP16 Baseline | 16.0 | Reference |
| OIL4 | 4.0 | +0.1 PPL |
| OIL8 (65536 centroids) | 0.91 | +1.5 PPL |
| OIL8 (256 centroids) | 0.85 | +2.5 PPL |

## Implementation

These research papers directly inform:
- `oil_format.h/cpp` — OIL binary format
- `kernel_oil4.cpp`, `kernel_oil8.cpp` — Quantization kernels
- `ste_quantizer.h/cpp` — Differentiable quantization
- `codebook.h/cpp` — Codebook training
- `format_planner.h/cpp` — Format assignment
- `moe.h`, `moe_enhance.h` — MoE routing
- `engines/OIL8/` — OIL8 codec
