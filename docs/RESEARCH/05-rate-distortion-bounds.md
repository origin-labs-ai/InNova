# Research Phase 5: Rate-Distortion Bounds for NN Weight Compression

## Theoretical Limits of Weight Compression

### Rate-Distortion Function (Shannon, 1959)

For a source X with distortion measure d(X,X̂):
```
R(D) = min_{p(x̂|x): E[d(X,X̂)] ≤ D} I(X; X̂)
```

This gives the MINIMUM bits per weight needed to achieve distortion ≤ D.

### Gaussian Source Bound

For i.i.d. Gaussian weights with variance σ²:
```
R(D) = ½ log₂(σ²/D)    for 0 ≤ D ≤ σ²
```

**Implication for FP32→INT8 (OIL8):**
- FP32 has 23 mantissa bits ≈ 7 decimal digits precision
- INT8 (256 levels) has ~8 bits entropy
- To achieve D→0 (lossless), R→∞ — IMPOSSIBLE with finite bits

### Heavy-Tailed Distribution Reality

Neural network weights follow **heavy-tailed distributions** (Mahoney & Martin, 2019), NOT Gaussian:
- Most weights near zero (high density)
- Some weights very large (long tails)
- Empirically: approximately Laplace or power-law

**Key finding from NWC paper (arXiv:2510.11234):**
- Gaussian-optimized codecs perform poorly on heavy-tailed data at mid-to-high bitrates
- The relative error for Laplace distribution grows much higher than Gaussian as bitrate increases
- This is because Gaussian codecs don't assign enough centroids to the tails

### Rate-Distortion for NN Weights (Isik et al., 2022)

Paper: "An Information-Theoretic Justification for Model Pruning" (AISTATS 2022)

Key results:
1. Normalized weights u = w/||w||₁ follow approximately uniform distribution on simplex
2. Rate-distortion function derived for this setting
3. **Successive refinement** (progressive coding) can achieve optimal R-D tradeoff
4. Layer-wise compression is suboptimal — joint compression across layers is better

### Bits-Per-Weight (BPW) Analysis for OIL Formats

**OIL8 (256 centroids):**
- 8 bits per weight index
- Plus codebook overhead: 256 × 32 bits per block = 8192 bits overhead
- For block size 128 weights: 8 bits/weight + 64 bits overhead = ~8.5 BPW effective
- Compared to FP32 (32 BPW): 3.76× compression
- **Distortion: NON-ZERO** — quantization error exists

**OIL4 (16 centroids):**
- 4 bits per weight index
- For block size 128: 4 bits/weight + ~32 bits overhead = ~4.25 BPW
- **Distortion: LARGER** than OIL8

**Ternary (I2_S):**
- 2 bits per weight
- **Distortion: LARGEST** among the three

### Mixed Format (OIL8 + OIL4 + Ternary)

Per-block adaptive assignment:
- Critical layers: OIL8 (8 BPW) — small distortion
- Medium layers: OIL4 (4 BPW) — moderate distortion
- Redundant layers: Ternary (2 BPW) — large distortion

**Average BPW:** Depends on allocation ratio:
- 1% OIL8 + 4% OIL4 + 95% ternary → ~2.12 BPW (current static split)
- Optimal allocation could achieve lower BPW with acceptable distortion

### Fundamental Limit

**No compression scheme can achieve 0% loss (D=0) with finite bits for continuous-valued weights.**

The ONLY paths to "effectively 0% loss" for the OIL format:
1. **Train-in-format:** Model learns to produce correct outputs given quantized weights
2. **Residual refinement:** Store quantization error separately (adds bits but achieves lossless)
3. **Entropy coding on top:** Huffman/arithmetic coding of quantized indices (lossless storage of the quantized representation)

**Option 1 is most promising for OIL:** The model trains with quantization in the loop, so the final model's performance is what it would be with those exact weights. This is "lossless inference" — not mathematical losslessness, but practical equivalence.
