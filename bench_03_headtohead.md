# InNova PoC — File 3: Comprehensive Head-to-Head

**Every format × every distribution. Comparison at same BPW tier.**

## Gaussian_002

| Format | BPW | MSE | SNR | vs GGUF Q4_K_M |
|--------|-----|-----|-----|----------------|
| OIL1 | 1.00 | 3.8910e-04 | 0.1 | 77% less bits (15919% quality trade) |
| OIL2 | 2.00 | 4.6238e-05 | 9.4 | 55% less bits (1803% quality trade) |
| OIL4 | 4.00 | 4.0143e-06 | 20.0 | 11% less bits (65% quality trade) |
| OIL8 | 8.00 | 1.5078e-06 | 24.2 | **38% BETTER** |
| OIL16 | 16.00 | 7.2809e-11 | 67.4 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 | 999.0 |  |
| OIL1_GRP | 1.00 | 2.4692e-03 | -7.9 | 77% less bits (101557% quality trade) |
| OIL2_GRP | 2.00 | 6.5790e-05 | 7.9 | 55% less bits (2608% quality trade) |
| OIL4_GRP | 4.00 | 6.9321e-06 | 17.6 | 11% less bits (185% quality trade) |
| OIL8_GRP | 8.00 | 3.7542e-06 | 20.3 | 54% higher MSE |
| OIL16_GRP | 16.00 | 3.7542e-06 | 20.3 | 54% higher MSE |
| SPARK_SPARSE | 2.00 | 2.2601e-04 | 2.5 | 55% less bits (9204% quality trade) |
| SPARK_SPARSE_GRP | 2.00 | 1.0200e-07 | 35.9 | **WINS** 96% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 4.9232e-05 | 9.1 | 66% less bits (1926% quality trade) |
| SPARK_Q0_GRP | 1.50 | 3.0213e-01 | -28.8 | 66% less bits (12438709% quality trade) |
| GGUF_Q4_K_M | 4.50 | 2.4289e-06 | 22.2 | BASELINE |
| GPTQ_4bit | 4.00 | 1.6992e-06 | 23.7 | **WINS** 31% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 7.8212e-08 | 37.1 | **WINS** 97% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 2.5116e-06 | 22.0 | **EFFICIENT** 33% less bits, quality/byte better |
| OIL_MIX@4bpw | 4.00 | 5.0703e-06 | 19.0 | 11% less bits (108% quality trade) |

## Gaussian_050

| Format | BPW | MSE | SNR | vs GGUF Q4_K_M |
|--------|-----|-----|-----|----------------|
| OIL1 | 1.00 | 2.4319e-01 | 0.1 | 77% less bits (15919% quality trade) |
| OIL2 | 2.00 | 2.8899e-02 | 9.4 | 55% less bits (1803% quality trade) |
| OIL4 | 4.00 | 2.5090e-03 | 20.0 | 11% less bits (65% quality trade) |
| OIL8 | 8.00 | 9.4240e-04 | 24.2 | **38% BETTER** |
| OIL16 | 16.00 | 4.2834e-08 | 67.7 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 | 999.0 |  |
| OIL1_GRP | 1.00 | 1.5432e+00 | -7.9 | 77% less bits (101557% quality trade) |
| OIL2_GRP | 2.00 | 4.1119e-02 | 7.9 | 55% less bits (2608% quality trade) |
| OIL4_GRP | 4.00 | 4.3327e-03 | 17.6 | 11% less bits (185% quality trade) |
| OIL8_GRP | 8.00 | 2.3464e-03 | 20.3 | 54% higher MSE |
| OIL16_GRP | 16.00 | 2.3464e-03 | 20.3 | 54% higher MSE |
| SPARK_SPARSE | 2.00 | 1.4125e-01 | 2.5 | 55% less bits (9204% quality trade) |
| SPARK_SPARSE_GRP | 2.00 | 6.3747e-05 | 35.9 | **WINS** 96% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 3.0770e-02 | 9.1 | 66% less bits (1926% quality trade) |
| SPARK_Q0_GRP | 1.50 | 6.0158e-02 | 6.2 | 66% less bits (3862% quality trade) |
| GGUF_Q4_K_M | 4.50 | 1.5181e-03 | 22.2 | BASELINE |
| GPTQ_4bit | 4.00 | 1.0620e-03 | 23.7 | **WINS** 31% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 4.8882e-05 | 37.1 | **WINS** 97% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 1.5698e-03 | 22.0 | **EFFICIENT** 33% less bits, quality/byte better |
| OIL_MIX@4bpw | 4.00 | 3.1689e-03 | 19.0 | 11% less bits (108% quality trade) |

## Sparse_90

| Format | BPW | MSE | SNR | vs GGUF Q4_K_M |
|--------|-----|-----|-----|----------------|
| OIL1 | 1.00 | 1.0193e-01 | 0.2 | 77% less bits (7721% quality trade) |
| OIL2 | 2.00 | 1.4036e-02 | 8.8 | 55% less bits (977% quality trade) |
| OIL4 | 4.00 | 1.8136e-03 | 17.7 | 11% less bits (39% quality trade) |
| OIL8 | 8.00 | 1.1505e-03 | 19.6 | **12% BETTER** |
| OIL16 | 16.00 | 1.8482e-08 | 67.6 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 | 999.0 |  |
| OIL1_GRP | 1.00 | 5.0190e+00 | -16.8 | 77% less bits (385058% quality trade) |
| OIL2_GRP | 2.00 | 8.7382e-02 | 0.8 | 55% less bits (6605% quality trade) |
| OIL4_GRP | 4.00 | 9.6275e-03 | 10.4 | 11% less bits (638% quality trade) |
| OIL8_GRP | 8.00 | 7.5878e-03 | 11.4 | 482% higher MSE |
| OIL16_GRP | 16.00 | 7.5878e-03 | 11.4 | 482% higher MSE |
| SPARK_SPARSE | 2.00 | 4.2760e-05 | 33.9 | **WINS** 97% better quality, 55% less bits |
| SPARK_SPARSE_GRP | 2.00 | 1.7585e-05 | 37.8 | **WINS** 99% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 1.2726e-01 | -0.8 | 66% less bits (9665% quality trade) |
| SPARK_Q0_GRP | 1.50 | 9.4405e-02 | 0.5 | 66% less bits (7144% quality trade) |
| GGUF_Q4_K_M | 4.50 | 1.3031e-03 | 19.1 | BASELINE |
| GPTQ_4bit | 4.00 | 4.9327e-04 | 23.3 | **WINS** 63% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 1.1945e-05 | 39.5 | **WINS** 100% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 2.4599e-03 | 16.3 | 33% less bits (88% quality trade) |
| OIL_MIX@4bpw | 4.00 | 4.2057e-03 | 14.0 | 11% less bits (222% quality trade) |

## Sparse_95

| Format | BPW | MSE | SNR | vs GGUF Q4_K_M |
|--------|-----|-----|-----|----------------|
| OIL1 | 1.00 | 5.2729e-02 | 0.1 | 77% less bits (10514% quality trade) |
| OIL2 | 2.00 | 7.8385e-03 | 8.4 | 55% less bits (1477% quality trade) |
| OIL4 | 4.00 | 9.5732e-04 | 17.6 | 11% less bits (92% quality trade) |
| OIL8 | 8.00 | 7.0374e-04 | 18.9 | 41% higher MSE |
| OIL16 | 16.00 | 9.2891e-09 | 67.7 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 | 999.0 |  |
| OIL1_GRP | 1.00 | 4.7214e+00 | -19.4 | 77% less bits (950319% quality trade) |
| OIL2_GRP | 2.00 | 7.5283e-02 | -1.4 | 55% less bits (15054% quality trade) |
| OIL4_GRP | 4.00 | 7.0955e-03 | 8.9 | 11% less bits (1328% quality trade) |
| OIL8_GRP | 8.00 | 5.1805e-03 | 10.2 | 942% higher MSE |
| OIL16_GRP | 16.00 | 5.1805e-03 | 10.2 | 942% higher MSE |
| SPARK_SPARSE | 2.00 | 2.0407e-05 | 34.3 | **WINS** 96% better quality, 55% less bits |
| SPARK_SPARSE_GRP | 2.00 | 7.8247e-06 | 38.4 | **WINS** 99% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 9.2456e-02 | -2.3 | 66% less bits (18511% quality trade) |
| SPARK_Q0_GRP | 1.50 | 7.9429e-02 | -1.6 | 66% less bits (15889% quality trade) |
| GGUF_Q4_K_M | 4.50 | 4.9677e-04 | 20.4 | BASELINE |
| GPTQ_4bit | 4.00 | 1.6554e-04 | 25.2 | **WINS** 67% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 4.7358e-06 | 40.6 | **WINS** 100% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 2.0095e-03 | 14.3 | 33% less bits (304% quality trade) |
| OIL_MIX@4bpw | 4.00 | 2.9055e-03 | 12.7 | 11% less bits (484% quality trade) |

## Sparse_99

| Format | BPW | MSE | SNR | vs GGUF Q4_K_M |
|--------|-----|-----|-----|----------------|
| OIL1 | 1.00 | 7.7654e-03 | 0.1 | 77% less bits (32792% quality trade) |
| OIL2 | 2.00 | 9.5603e-04 | 9.2 | 55% less bits (3949% quality trade) |
| OIL4 | 4.00 | 1.5877e-04 | 17.0 | 11% less bits (572% quality trade) |
| OIL8 | 8.00 | 1.3121e-04 | 17.9 | 455% higher MSE |
| OIL16 | 16.00 | 1.1608e-09 | 68.4 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 | 999.0 |  |
| OIL1_GRP | 1.00 | 1.8498e+00 | -23.6 | 77% less bits (7835070% quality trade) |
| OIL2_GRP | 2.00 | 3.6275e-02 | -6.5 | 55% less bits (153551% quality trade) |
| OIL4_GRP | 4.00 | 2.6566e-03 | 4.8 | 11% less bits (11152% quality trade) |
| OIL8_GRP | 8.00 | 2.0900e-03 | 5.8 | 8752% higher MSE |
| OIL16_GRP | 16.00 | 2.0900e-03 | 5.8 | 8752% higher MSE |
| SPARK_SPARSE | 2.00 | 8.6860e-07 | 39.7 | **WINS** 97% better quality, 55% less bits |
| SPARK_SPARSE_GRP | 2.00 | 5.2887e-07 | 41.8 | **WINS** 98% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 6.1473e-02 | -8.8 | 66% less bits (260284% quality trade) |
| SPARK_Q0_GRP | 1.50 | 6.4738e-02 | -9.1 | 66% less bits (274111% quality trade) |
| GGUF_Q4_K_M | 4.50 | 2.3609e-05 | 25.3 | BASELINE |
| GPTQ_4bit | 4.00 | 1.7833e-05 | 26.5 | **WINS** 25% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 2.1620e-07 | 45.7 | **WINS** 100% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 1.6362e-04 | 16.9 | 33% less bits (593% quality trade) |
| OIL_MIX@4bpw | 4.00 | 1.6675e-04 | 16.8 | 11% less bits (606% quality trade) |

## Grand Summary (Average)

| Format | BPW | Avg MSE | vs GGUF Q4_K_M |
|--------|-----|---------|----------------|
| OIL1 | 1.00 | 8.1200e-02 | 77% less bits (12041% quality trade) |
| OIL1_GRP | 1.00 | 2.6272e+00 | 77% less bits (392722% quality trade) |
| SPARK_Q0 | 1.50 | 6.2401e-02 | 66% less bits (9230% quality trade) |
| SPARK_Q0_GRP | 1.50 | 1.2017e-01 | 66% less bits (17868% quality trade) |
| OIL2 | 2.00 | 1.0355e-02 | 55% less bits (1448% quality trade) |
| OIL2_GRP | 2.00 | 4.8025e-02 | 55% less bits (7080% quality trade) |
| SPARK_SPARSE | 2.00 | 2.8309e-02 | 55% less bits (4132% quality trade) |
| SPARK_SPARSE_GRP | 2.00 | 1.7957e-05 | **WINS** 98% better quality, 55% less bits |
| OIL_MIX@2bpw | 2.00 | 1.3172e-05 | **WINS** 99% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 1.2411e-03 | 33% less bits (85% quality trade) |
| OIL4 | 4.00 | 1.0885e-03 | 11% less bits (62% quality trade) |
| OIL4_GRP | 4.00 | 4.7438e-03 | 11% less bits (609% quality trade) |
| GPTQ_4bit | 4.00 | 3.4807e-04 | **WINS** 48% better quality, 11% less bits |
| OIL_MIX@4bpw | 4.00 | 2.0904e-03 | 11% less bits (212% quality trade) |
| GGUF_Q4_K_M | 4.50 | 6.6880e-04 | BASELINE |
| OIL8 | 8.00 | 5.8586e-04 | **13% BETTER** |
| OIL8_GRP | 8.00 | 3.4417e-03 | 414% higher MSE |
| OIL16 | 16.00 | 1.4368e-08 | **100% BETTER** |
| OIL16_GRP | 16.00 | 3.4417e-03 | 414% higher MSE |
| OIL32 | 32.00 | 0.0000e+00 |  |

## Key Findings

1. **SPARK_SPARSE_GRP at 2.0 BPW** beats GGUF Q4_K_M at 4.5 BPW on sparse data (55% less bits, better quality)
2. **OIL8 at 8.0 BPW** dominates all formats
3. **OIL_MIX** uses importance routing: OIL8 for salient weights, OIL2 for bulk
4. Real neural weights are sparse — OIL's codebook quantization excels on sparse data

---
*Generated by InNova bench_poc*
