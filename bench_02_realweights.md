# InNova PoC — File 2: Real Neural Weight Distributions

**Distributions:** Sparse (90%, 95%, 99%) — mimics real transformer weights

## Sparse_90

| Format | BPW | MSE | vs GGUF Q4_K_M |
|--------|-----|-----|----------------|
| OIL1 | 1.00 | 1.0193e-01 | 77% less bits (7721% quality trade) |
| OIL2 | 2.00 | 1.4036e-02 | 55% less bits (977% quality trade) |
| OIL4 | 4.00 | 1.8136e-03 | 11% less bits (39% quality trade) |
| OIL8 | 8.00 | 1.1505e-03 | **12% BETTER** |
| OIL16 | 16.00 | 1.8482e-08 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 |  |
| OIL1_GRP | 1.00 | 5.0190e+00 | 77% less bits (385058% quality trade) |
| OIL2_GRP | 2.00 | 8.7382e-02 | 55% less bits (6605% quality trade) |
| OIL4_GRP | 4.00 | 9.6275e-03 | 11% less bits (638% quality trade) |
| OIL8_GRP | 8.00 | 7.5878e-03 | 482% higher MSE |
| OIL16_GRP | 16.00 | 1.0550e-01 | 7995% higher MSE |
| SPARK_SPARSE | 2.00 | 4.2760e-05 | **WINS** 97% better quality, 55% less bits |
| SPARK_SPARSE_GRP | 2.00 | 1.7585e-05 | **WINS** 99% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 1.2726e-01 | 66% less bits (9665% quality trade) |
| SPARK_Q0_GRP | 1.50 | 9.4405e-02 | 66% less bits (7144% quality trade) |
| GGUF_Q4_K_M | 4.50 | 1.3031e-03 | BASELINE |
| GPTQ_4bit | 4.00 | 4.9327e-04 | **WINS** 63% better quality, 11% less bits |
| OIL4_CW | 4.00 | 4.7392e-04 | **WINS** 64% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 1.1945e-05 | **WINS** 100% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 2.4599e-03 | 33% less bits (88% quality trade) |
| OIL_MIX@4bpw | 4.00 | 4.2057e-03 | 11% less bits (222% quality trade) |

## Sparse_95

| Format | BPW | MSE | vs GGUF Q4_K_M |
|--------|-----|-----|----------------|
| OIL1 | 1.00 | 5.2729e-02 | 77% less bits (10514% quality trade) |
| OIL2 | 2.00 | 7.8385e-03 | 55% less bits (1477% quality trade) |
| OIL4 | 4.00 | 9.5732e-04 | 11% less bits (92% quality trade) |
| OIL8 | 8.00 | 7.0374e-04 | 41% higher MSE |
| OIL16 | 16.00 | 9.2891e-09 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 |  |
| OIL1_GRP | 1.00 | 4.7214e+00 | 77% less bits (950319% quality trade) |
| OIL2_GRP | 2.00 | 7.5283e-02 | 55% less bits (15054% quality trade) |
| OIL4_GRP | 4.00 | 7.0955e-03 | 11% less bits (1328% quality trade) |
| OIL8_GRP | 8.00 | 5.1805e-03 | 942% higher MSE |
| OIL16_GRP | 16.00 | 5.4427e-02 | 10856% higher MSE |
| SPARK_SPARSE | 2.00 | 2.0407e-05 | **WINS** 96% better quality, 55% less bits |
| SPARK_SPARSE_GRP | 2.00 | 7.8247e-06 | **WINS** 99% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 9.2456e-02 | 66% less bits (18511% quality trade) |
| SPARK_Q0_GRP | 1.50 | 7.9429e-02 | 66% less bits (15889% quality trade) |
| GGUF_Q4_K_M | 4.50 | 4.9677e-04 | BASELINE |
| GPTQ_4bit | 4.00 | 1.6554e-04 | **WINS** 67% better quality, 11% less bits |
| OIL4_CW | 4.00 | 3.3984e-04 | **WINS** 32% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 4.7358e-06 | **WINS** 100% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 2.0095e-03 | 33% less bits (304% quality trade) |
| OIL_MIX@4bpw | 4.00 | 2.9055e-03 | 11% less bits (484% quality trade) |

## Sparse_99

| Format | BPW | MSE | vs GGUF Q4_K_M |
|--------|-----|-----|----------------|
| OIL1 | 1.00 | 7.7654e-03 | 77% less bits (32792% quality trade) |
| OIL2 | 2.00 | 9.5603e-04 | 55% less bits (3949% quality trade) |
| OIL4 | 4.00 | 1.5877e-04 | 11% less bits (572% quality trade) |
| OIL8 | 8.00 | 1.3121e-04 | 455% higher MSE |
| OIL16 | 16.00 | 1.1608e-09 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 |  |
| OIL1_GRP | 1.00 | 1.8498e+00 | 77% less bits (7835070% quality trade) |
| OIL2_GRP | 2.00 | 3.6275e-02 | 55% less bits (153551% quality trade) |
| OIL4_GRP | 4.00 | 2.6566e-03 | 11% less bits (11152% quality trade) |
| OIL8_GRP | 8.00 | 2.0900e-03 | 8752% higher MSE |
| OIL16_GRP | 16.00 | 8.0276e-03 | 33902% higher MSE |
| SPARK_SPARSE | 2.00 | 8.6860e-07 | **WINS** 97% better quality, 55% less bits |
| SPARK_SPARSE_GRP | 2.00 | 5.2887e-07 | **WINS** 98% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 6.1473e-02 | 66% less bits (260284% quality trade) |
| SPARK_Q0_GRP | 1.50 | 6.4738e-02 | 66% less bits (274111% quality trade) |
| GGUF_Q4_K_M | 4.50 | 2.3609e-05 | BASELINE |
| GPTQ_4bit | 4.00 | 1.7833e-05 | **WINS** 25% better quality, 11% less bits |
| OIL4_CW | 4.00 | 1.6323e-04 | 11% less bits (591% quality trade) |
| OIL_MIX@2bpw | 2.00 | 2.1620e-07 | **WINS** 100% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 1.6362e-04 | 33% less bits (593% quality trade) |
| OIL_MIX@4bpw | 4.00 | 1.6675e-04 | 11% less bits (606% quality trade) |

## Attn_QKV

| Format | BPW | MSE | vs GGUF Q4_K_M |
|--------|-----|-----|----------------|
| OIL1 | 1.00 | 3.8910e-04 | 77% less bits (15919% quality trade) |
| OIL2 | 2.00 | 4.6238e-05 | 55% less bits (1803% quality trade) |
| OIL4 | 4.00 | 4.0143e-06 | 11% less bits (65% quality trade) |
| OIL8 | 8.00 | 1.5078e-06 | **38% BETTER** |
| OIL16 | 16.00 | 7.2809e-11 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 |  |
| OIL1_GRP | 1.00 | 2.4692e-03 | 77% less bits (101557% quality trade) |
| OIL2_GRP | 2.00 | 6.5790e-05 | 55% less bits (2608% quality trade) |
| OIL4_GRP | 4.00 | 6.9321e-06 | 11% less bits (185% quality trade) |
| OIL8_GRP | 8.00 | 3.7542e-06 | 54% higher MSE |
| OIL16_GRP | 16.00 | 4.0074e-04 | 16398% higher MSE |
| SPARK_SPARSE | 2.00 | 2.2601e-04 | 55% less bits (9204% quality trade) |
| SPARK_SPARSE_GRP | 2.00 | 1.0200e-07 | **WINS** 96% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 4.9232e-05 | 66% less bits (1926% quality trade) |
| SPARK_Q0_GRP | 1.50 | 3.0213e-01 | 66% less bits (12438709% quality trade) |
| GGUF_Q4_K_M | 4.50 | 2.4289e-06 | BASELINE |
| GPTQ_4bit | 4.00 | 1.6992e-06 | **WINS** 31% better quality, 11% less bits |
| OIL4_CW | 4.00 | 1.5576e-06 | **WINS** 36% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 7.8212e-08 | **WINS** 97% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 2.5116e-06 | **EFFICIENT** 33% less bits, quality/byte better |
| OIL_MIX@4bpw | 4.00 | 5.0703e-06 | 11% less bits (108% quality trade) |

## FFN_Down

| Format | BPW | MSE | vs GGUF Q4_K_M |
|--------|-----|-----|----------------|
| OIL1 | 1.00 | 6.0797e-04 | 77% less bits (15919% quality trade) |
| OIL2 | 2.00 | 7.2247e-05 | 55% less bits (1803% quality trade) |
| OIL4 | 4.00 | 6.2724e-06 | 11% less bits (65% quality trade) |
| OIL8 | 8.00 | 2.3560e-06 | **38% BETTER** |
| OIL16 | 16.00 | 1.0685e-10 | **100% BETTER** |
| OIL32 | 32.00 | 0.0000e+00 |  |
| OIL1_GRP | 1.00 | 3.8581e-03 | 77% less bits (101557% quality trade) |
| OIL2_GRP | 2.00 | 1.0280e-04 | 55% less bits (2608% quality trade) |
| OIL4_GRP | 4.00 | 1.0832e-05 | 11% less bits (185% quality trade) |
| OIL8_GRP | 8.00 | 5.8659e-06 | 54% higher MSE |
| OIL16_GRP | 16.00 | 6.2615e-04 | 16398% higher MSE |
| SPARK_SPARSE | 2.00 | 3.5313e-04 | 55% less bits (9204% quality trade) |
| SPARK_SPARSE_GRP | 2.00 | 1.5937e-07 | **WINS** 96% better quality, 55% less bits |
| SPARK_Q0 | 1.50 | 7.6926e-05 | 66% less bits (1926% quality trade) |
| SPARK_Q0_GRP | 1.50 | 2.9723e-01 | 66% less bits (7831585% quality trade) |
| GGUF_Q4_K_M | 4.50 | 3.7952e-06 | BASELINE |
| GPTQ_4bit | 4.00 | 2.6550e-06 | **WINS** 31% better quality, 11% less bits |
| OIL4_CW | 4.00 | 2.4337e-06 | **WINS** 36% better quality, 11% less bits |
| OIL_MIX@2bpw | 2.00 | 1.2221e-07 | **WINS** 97% better quality, 55% less bits |
| OIL_MIX@3bpw | 3.00 | 3.9244e-06 | **EFFICIENT** 33% less bits, quality/byte better |
| OIL_MIX@4bpw | 4.00 | 7.9224e-06 | 11% less bits (108% quality trade) |


---
*Generated by InNova bench_poc*
