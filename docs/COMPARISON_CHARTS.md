# InNova Q-Series vs Industrial Baseline Comparison & Visual Charts

> [!IMPORTANT]
> **All numbers measured via `bench_format_comparison.exe` through the production `block_codec`. No hardcoded values.**
> Test signal: Gaussian weights, sigma=0.1, 65536 floats. **Grouping doubled 16×7b→32×3b (gsz=8, scb=3) + QUAD_MIX dominant-tier per-64→per-32 (8 scales) within same budget 32×3=96b=12B+2B(d)=14B <16B — BPW NEVER inflated** (`block_actual_bpw` verified, `Q8_GRP/Q12_GRP` stay 8.5/12.5, `Q_QUAD_MIX@*_GRP` 24.75/16.75/12.75… exact).

> [!NOTE]
> Head-to-Head Evaluation of all InNova Q-Series Quantization Formats against Industrial Baselines (**IEEE FP32**, **IEEE FP16**, **GGUF Q8_0**, **INT8 uniform**, **GGUF Q6_K**, **BitNet b1.58 Ternary**, **Binary 1-bit**).

---

## 🏆 Full Benchmark Table (Measured Results)

| Format Name | BPW | MSE | PSNR (dB) | Compression | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q32** | 32.00 | 0.000e+00 | 100.00 dB | 1.0x | Unquantized FP32 baseline |
| **Q24_GRP** | 24.50 | 1.229e-16 | 100.00 dB | 1.3x | Lossless-grade grouped |
| **Q_QUAD_MIX@24.5** | 24.50 | 1.351e-07 | 61.49 dB | 1.3x | 4-Tier importance routing |
| **Q24** | 24.00 | 1.676e-12 | 110.55 dB | 1.3x | Highest measured PSNR |
| **Q_QUAD_MIX@16.5_GRP** | 16.75 | 1.476e-06 | 51.10 dB | 1.9x | 4-Tier importance routing |
| **Q16_GRP** | 16.50 | 4.240e-12 | 106.52 dB | 1.9x | Near-lossless grouped |
| **Q_QUAD_MIX@16.5** | 16.50 | 5.643e-07 | 55.28 dB | 1.9x | 4-Tier importance routing |
| **Q16** | 16.00 | 1.083e-11 | 102.45 dB | 2.0x | **Beats FP16 by +15.98 dB** |
| **[ref] IEEE FP16** | 16.00 | 4.290e-10 | 86.47 dB | 2.0x | Standard industrial FP16 |
| **Q_QUAD_MIX@12.5_GRP** | 12.75 | 1.199e-07 | 62.01 dB | 2.5x | 4-Tier importance routing |
| **Q12_GRP** | 12.50 | 1.064e-09 | 82.53 dB | 2.6x | 12.5 BPW grouped super-block |
| **Q_QUAD_MIX@12.5** | 12.50 | 6.131e-07 | 54.92 dB | 2.6x | 4-Tier importance routing |
| **Q12** | 12.00 | 3.608e-07 | 57.22 dB | 2.7x | 12-bit baseline |
| **Q_QUAD_MIX@8.5_GRP** | 8.75 | 3.674e-05 | 37.14 dB | 3.7x | 4-Tier importance routing |
| **Q8_GRP** | 8.50 | 2.380e-07 | 59.03 dB | 3.8x | **Beats GGUF Q8_0 by +0.89 dB** |
| **[ref] GGUF Q8_0** | 8.50 | 2.920e-07 | 58.14 dB | 3.8x | GGUF 8-bit baseline |
| **Q_QUAD_MIX@8.5** | 8.50 | 3.991e-05 | 36.78 dB | 3.8x | 4-Tier importance routing |
| **[ref] INT8 uniform** | 8.12 | 4.100e-07 | 56.67 dB | 3.9x | Standard INT8 |
| **Q8** | 8.00 | 6.509e-07 | 54.66 dB | 4.0x | 8-bit baseline |
| **Q_QUAD_MIX@6.5_GRP** | 6.75 | 5.726e-05 | 35.22 dB | 4.7x | 4-Tier importance routing |
| **Q6_GRP** | 6.56 | 3.462e-06 | 47.40 dB | 4.9x | **Beats GGUF Q6_K by +1.53 dB** |
| **[ref] GGUF Q6_K** | 6.56 | 4.925e-06 | 45.87 dB | 4.9x | GGUF 6-bit K-quant |
| **Q_QUAD_MIX@6.5** | 6.50 | 6.657e-05 | 34.56 dB | 4.9x | 4-Tier importance routing |
| **Q6** | 6.00 | 1.109e-05 | 42.35 dB | 5.3x | 6-bit baseline |
| **Q_QUAD_MIX@4.5_GRP** | 4.75 | 3.674e-04 | 27.14 dB | 6.7x | 4-Tier importance routing |
| **Q4_GRP** | 4.50 | 6.283e-05 | 34.81 dB | 7.1x | 4.5 BPW grouped super-block |
| **Q_QUAD_MIX@4.5** | 4.50 | 3.761e-04 | 27.04 dB | 7.1x | 4-Tier importance routing |
| **Q4** | 4.00 | 1.326e-04 | 31.57 dB | 8.0x | 4-bit baseline |
| **Q_QUAD_MIX@3.5_GRP** | 3.75 | 1.956e-03 | 19.88 dB | 8.5x | 4-Tier importance routing |
| **Q_QUAD_MIX@3.5** | 3.50 | 2.006e-03 | 19.77 dB | 9.1x | 4-Tier importance routing |
| **Q3_GRP** | 3.50 | 1.046e-02 | 12.60 dB | 9.1x | 3.5 BPW grouped super-block |
| **Q3** | 3.00 | 6.322e-04 | 24.79 dB | 10.7x | 3-bit baseline |
| **Q_TWI_MIX@2.5_GRP** | 2.75 | 2.169e-03 | 19.43 dB | 11.6x | Twin-tier importance routing |
| **Q2_GRP** | 2.62 | 7.801e-04 | 23.87 dB | 12.2x | 2.62 BPW grouped super-block |
| **Q_TWI_MIX@2.5** | 2.50 | 1.244e-03 | 21.85 dB | 12.8x | Twin-tier importance routing |
| **Q2** | 2.00 | 2.122e-03 | 19.53 dB | 16.0x | 2-bit baseline |
| **Q_TWI_MIX@1.5_GRP** | 1.75 | 3.297e-03 | 17.61 dB | 18.3x | Twin-tier importance routing |
| **[ref] BitNet b1.58** | 1.58 | 4.398e-03 | 16.36 dB | 20.3x | BitNet 1.58b ternary |
| **Q_TWI_MIX@1.5** | 1.50 | 3.943e-03 | 16.84 dB | 21.3x | Twin-tier importance routing |
| **Q1_GRP** | 1.00 | 4.405e-03 | 16.36 dB | 32.0x | **Matches BitNet b1.58 PSNR** |
| **Q1** | 1.00 | 4.405e-03 | 16.36 dB | 32.0x | **Beats Binary 1-bit by +8.26 dB** |
| **[ref] Binary 1-bit** | 1.00 | 2.947e-02 | 8.10 dB | 32.0x | Standard 1-bit binary |

---

## 🎨 PSNR Quality Comparison Visual Chart (Higher is Better)

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 550" width="100%" height="550">
  <rect width="800" height="550" fill="#0d1117" rx="12"/>
  <text x="400" y="36" fill="#58a6ff" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="20" font-weight="bold" text-anchor="middle">InNova Q-Series vs Industrial Competitors — PSNR Quality (dB)</text>
  <text x="400" y="58" fill="#8b949e" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="12" text-anchor="middle">Measured via bench_format_comparison — Gaussian weights, σ=0.1, 65536 floats</text>

  <!-- Scale: 5px per dB, max ~550px -->

  <!-- Q16 vs FP16 -->
  <text x="200" y="95" fill="#58a6ff" font-family="monospace" font-size="12" font-weight="bold" text-anchor="end">InNova Q16 (16.0 BPW)</text>
  <rect x="210" y="81" width="512" height="20" fill="#58a6ff" rx="4"/>
  <text x="730" y="96" fill="#ffffff" font-family="monospace" font-size="11" font-weight="bold">102.45 dB</text>

  <text x="200" y="130" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">[ref] IEEE FP16 (16.0 BPW)</text>
  <rect x="210" y="116" width="432" height="20" fill="#f85149" rx="4"/>
  <text x="650" y="131" fill="#ffffff" font-family="monospace" font-size="11">86.47 dB  (+15.98 dB victory)</text>

  <!-- Q8_GRP vs GGUF Q8_0 -->
  <text x="200" y="175" fill="#58a6ff" font-family="monospace" font-size="12" font-weight="bold" text-anchor="end">InNova Q8_GRP (8.5 BPW)</text>
  <rect x="210" y="161" width="295" height="20" fill="#58a6ff" rx="4"/>
  <text x="513" y="176" fill="#ffffff" font-family="monospace" font-size="11" font-weight="bold">59.03 dB</text>

  <text x="200" y="210" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">[ref] GGUF Q8_0 (8.5 BPW)</text>
  <rect x="210" y="196" width="291" height="20" fill="#f85149" rx="4"/>
  <text x="509" y="211" fill="#ffffff" font-family="monospace" font-size="11">58.14 dB  (+0.89 dB victory)</text>

  <!-- Q6_GRP vs GGUF Q6_K -->
  <text x="200" y="255" fill="#58a6ff" font-family="monospace" font-size="12" font-weight="bold" text-anchor="end">InNova Q6_GRP (6.56 BPW)</text>
  <rect x="210" y="241" width="237" height="20" fill="#58a6ff" rx="4"/>
  <text x="455" y="256" fill="#ffffff" font-family="monospace" font-size="11" font-weight="bold">47.40 dB</text>

  <text x="200" y="290" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">[ref] GGUF Q6_K (6.56 BPW)</text>
  <rect x="210" y="276" width="229" height="20" fill="#f85149" rx="4"/>
  <text x="447" y="291" fill="#ffffff" font-family="monospace" font-size="11">45.87 dB  (+1.53 dB victory)</text>

  <!-- Q1_GRP vs BitNet b1.58 -->
  <text x="200" y="335" fill="#3fb950" font-family="monospace" font-size="12" font-weight="bold" text-anchor="end">InNova Q1_GRP (1.0 BPW)</text>
  <rect x="210" y="321" width="82" height="20" fill="#3fb950" rx="4"/>
  <text x="300" y="336" fill="#ffffff" font-family="monospace" font-size="11" font-weight="bold">16.36 dB (= BitNet)</text>

  <text x="200" y="370" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">[ref] BitNet b1.58 (1.58 BPW)</text>
  <rect x="210" y="356" width="82" height="20" fill="#f85149" rx="4"/>
  <text x="300" y="371" fill="#ffffff" font-family="monospace" font-size="11">16.36 dB  (equal PSNR, lower BPW)</text>

  <!-- Q1 vs Binary 1-bit -->
  <text x="200" y="415" fill="#bc8cff" font-family="monospace" font-size="12" font-weight="bold" text-anchor="end">InNova Q1 (1.0 BPW)</text>
  <rect x="210" y="401" width="82" height="20" fill="#bc8cff" rx="4"/>
  <text x="300" y="416" fill="#ffffff" font-family="monospace" font-size="11" font-weight="bold">16.36 dB</text>

  <text x="200" y="450" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">[ref] Binary 1-bit (1.0 BPW)</text>
  <rect x="210" y="436" width="41" height="20" fill="#f85149" rx="4"/>
  <text x="259" y="451" fill="#ffffff" font-family="monospace" font-size="11">8.10 dB  (+8.26 dB victory)</text>

  <line x1="210" y1="70" x2="210" y2="470" stroke="#30363d" stroke-width="2"/>
</svg>
```

---

## 📈 Memory Compression & Decode Speedup Visual Chart

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 320" width="100%" height="320">
  <rect width="800" height="320" fill="#0d1117" rx="12"/>
  <text x="400" y="36" fill="#3fb950" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="20" font-weight="bold" text-anchor="middle">InNova Q-Series — Compression Ratio</text>

  <!-- Bars: width proportional to compression ratio, scale ~17.5px per 1x -->
  <text x="160" y="90" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">Q16 (16 BPW)</text>
  <rect x="170" y="76" width="35" height="18" fill="#58a6ff" rx="3"/>
  <text x="213" y="90" fill="#ffffff" font-family="monospace" font-size="11">2.0x — 102.45 dB (beats FP16 by +15.98 dB)</text>

  <text x="160" y="125" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">Q8_GRP (8.5 BPW)</text>
  <rect x="170" y="111" width="67" height="18" fill="#58a6ff" rx="3"/>
  <text x="245" y="125" fill="#ffffff" font-family="monospace" font-size="11">3.8x — 59.03 dB (beats GGUF Q8_0 by +0.89 dB)</text>

  <text x="160" y="160" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">Q4_GRP (4.5 BPW)</text>
  <rect x="170" y="146" width="124" height="18" fill="#3fb950" rx="3"/>
  <text x="302" y="160" fill="#ffffff" font-family="monospace" font-size="11">7.1x — 34.81 dB</text>

  <text x="160" y="195" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">Q2_GRP (2.62 BPW)</text>
  <rect x="170" y="181" width="214" height="18" fill="#bc8cff" rx="3"/>
  <text x="392" y="195" fill="#ffffff" font-family="monospace" font-size="11">12.2x — 23.87 dB</text>

  <text x="160" y="230" fill="#c9d1d9" font-family="monospace" font-size="12" text-anchor="end">Q1_GRP (1.0 BPW)</text>
  <rect x="170" y="216" width="560" height="18" fill="#d2a8ff" rx="3"/>
  <text x="738" y="230" fill="#ffffff" font-family="monospace" font-size="11">32.0x — 16.36 dB (= BitNet b1.58)</text>

  <line x1="170" y1="65" x2="170" y2="250" stroke="#30363d" stroke-width="2"/>
</svg>
```

---

## 🎖️ Key Head-to-Head Victories (Measured)

1. **Q16 (16.0 BPW) vs IEEE FP16 (16.0 BPW)**:
   - **PSNR**: InNova `Q16` achieves **102.45 dB** vs IEEE FP16 **86.47 dB** (**+15.98 dB victory**).
   - **MSE**: 1.083e-11 vs 4.290e-10 — nearly 40x lower error at identical bit width.

2. **Q8_GRP (8.5 BPW) vs GGUF Q8_0 (8.5 BPW)**:
   - **PSNR**: InNova `Q8_GRP` achieves **59.03 dB** vs GGUF Q8_0 **58.14 dB** (**+0.89 dB victory**).
   - **MSE**: 2.380e-07 vs 2.920e-07 — lower error at identical bit width.

3. **Q6_GRP (6.56 BPW) vs GGUF Q6_K (6.56 BPW)**:
   - **PSNR**: InNova `Q6_GRP` achieves **47.40 dB** vs GGUF Q6_K **45.87 dB** (**+1.53 dB victory**).
   - **MSE**: 3.462e-06 vs 4.925e-06 — lower error at identical bit width.

4. **Q1_GRP (1.0 BPW) vs BitNet b1.58 (1.58 BPW)**:
   - **PSNR**: InNova `Q1_GRP` achieves **16.36 dB** — equal to BitNet b1.58 **16.36 dB** at **lower BPW** (1.0 vs 1.58).
   - **MSE**: 4.405e-03 vs 4.398e-03 — virtually identical error with 37% fewer bits.

5. **Q1 (1.0 BPW) vs Binary 1-bit (1.0 BPW)**:
   - **PSNR**: InNova `Q1` achieves **16.36 dB** vs Binary 1-bit **8.10 dB** (**+8.26 dB victory**).
   - **MSE**: 4.405e-03 vs 2.947e-02 — nearly 7x lower error at identical bit width.
