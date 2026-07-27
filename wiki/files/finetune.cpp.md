# `finetune.cpp` — Fine-Tuning Implementation

**Path:** `src/finetune.cpp`

Fine-tuning support: full fine-tune and quantized fine-tune.

## FineTune Methods

| Method | Description |
|--------|-------------|
| `full_finetune()` | Update all parameters (higher cost, full precision) |
| `quantized_finetune()` | Quantize + fine-tune in OIL format (target any BPW) |

## Quantized Fine-Tuning (Native OIL)

```
Model weights quantized to target BPW using Lloyd-Max codebooks.
Available formats: Binary(1.0), SPARK_Q0(1.5), OIL2(2), OIL4(4), OIL8(8), OIL16(16), OIL32(32)
Or use mixed-precision: FormatPlanner auto-selects 2-mix or 4-mix for optimal quality.
```

## Freezing

```cpp
// Freeze specified layers by name
finetune(model, {
    .freeze_layers = {"tok_embeddings", "norm", "lm_head"},
    .target_bpw = 2.0,
    .learning_rate = 1e-5,
});
```
