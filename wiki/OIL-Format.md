# OIL Format Specification

> **O**ptimized **I**nference **L**oader — a single self-contained binary format for model storage.

## Design Goals

- **Self-contained**: Everything in one file: weights, config, tokenizer, optimizer state
- **Zero-dependency**: Pure binary format, no protobuf/flatbuffers required
- **Quantization-native**: Weights stored directly in quantized format
- **Fast loading**: Memory-mapped loading for instant model startup
- **Forward compatible**: Versioned header allows format evolution

## Binary Layout

```
Offset      Size    Field
──────────────────────────────────
0x0000      4      Magic "OIL\0"
0x0004      4      Version (uint32)
0x0008      8      Number of tensors (uint64)
0x0010      8      Config section offset (uint64)
0x0018      8      Tensor metadata offset (uint64)
0x0020      8      Weight data offset (uint64)
0x0028      8      Tokenizer offset (uint64)
0x0030      8      Optimizer offset (uint64)
0x0038      8      Reserved for future use
──────────────────────────────────
0x0040      Config section (JSON string)
│   TransformerConfig as JSON
│
Tensor Metadata Table:
│   Each entry:
│     name (string, null-terminated)
│     shape (4 × int64)
│     rank (int32)
│     dtype (uint8: 0=I64, 1=U8, 2=U4, 3=I2, 4=I1, 5=F16, 6=F32)
│     data_offset (uint64)
│     data_size (uint64)
│
Weight Data Section:
│   Raw tensor data (quantized or FP)
│
Tokenizer Section (optional):
│   vocab_size (uint64)
│   vocab entries (string × vocab_size)
│   merge rules
│
Optimizer State (optional):
│   AdamW moments (m, v vectors)
│   step count
```

## Format Variants

MYTHOS.cpp defines **25 total formats**: 15 single-precision, 8 two-mix (twimix), and 2 four-mix.

| Category | Count | Examples |
|----------|-------|---------|
| Single-precision | 15 | OIL1–OIL32, SPARK_Q0, SPARK_SPARSE, all GRP variants |
| Two-mix (twimix) | 8 | Mixed BPW combinations using 2-format blending |
| Four-mix | 2 | Mixed BPW combinations using 4-format blending |

### Core Format Variants

| Variant | BPW | Storage | Centroids |
|---------|-----|---------|-----------|
| `OIL1` | 1.00 | 1 centroid per 32 weights | 1 (block mean) |
| `SPARK_Q0` | 1.50 | 2 bits/weight + per-block FP16 scale | 4 sign bins |
| `SPARK_SPARSE` | 2.00 | uint16 index + int8 value pairs | threshold sparsity |
| `OIL2` | 2.00 | 2 bits/weight | 4 centroids |
| `OIL4` | 4.00 | 4 bits/weight + 16×FP16 codebook | 16 centroids |
| `OIL8` | 8.00 | 8 bits/weight + 256×FP32 codebook | 256 centroids |
| `OIL16` | 16.0 | 16 bits/weight | — |
| `OIL32` | 32.0 | 32 bits/weight | FP32 lossless |

### GRP (Grouped) Variants

Most base formats have GRP variants (OIL2_GRP, OIL4_GRP, OIL8_GRP, SPARK_Q0_GRP, SPARK_SPARSE_GRP) that add per-group scale/zp. OIL1_GRP, OIL16_GRP, and OIL32_GRP are not implemented. GRP variants are lossy (improved quality via per-group scale/zp), not lossless.

## Quantization Format Details

### OIL2
```
Centroid table: 4 × FP32 = 16 bytes
Index array: 2 bits per weight (4 weights per byte)
Effective BPW: 2.0 (independent of n)
```

### OIL4
```
Centroid table: 16 × FP32 = 64 bytes
Index array: 4 bits per weight (2 weights per byte)
Effective BPW: 4.0 (independent of n)
```

### OIL8
```
Centroid table: 256 × FP32 = 1024 bytes
Index array: 8 bits per weight (1 weight per byte)
Effective BPW: 8.0 (independent of n)
```

## Reading & Writing

See [OIL Format Implementation](files/oil_format.cpp.md) for detailed code documentation.

```cpp
// Reading
OILReader reader("model.oil");
auto config = reader.read_config();
auto weights = reader.read_tensor("layers.0.attention.q.weight");

// Writing
OILWriter writer("model.oil", config);
writer.write_tensor("layers.0.attention.q.weight", weights);
writer.finalize();
```
