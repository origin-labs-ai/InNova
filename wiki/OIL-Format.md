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

MYTHOS.cpp defines **29 total formats**: 12 single-precision, 13 two-mix (twimix), and 4 four-mix.

| Category | Count | Examples |
|----------|-------|---------|
| Single-precision | 12 | OIL1–OIL32, Binary, Ternary, FP16, FP32, INT8, etc. |
| Two-mix (twimix) | 13 | Mixed BPW combinations using 2-format blending |
| Four-mix | 4 | Mixed BPW combinations using 4-format blending |

### Core Format Variants

| Variant | BPW | Storage | Centroids |
|---------|-----|---------|-----------|
| `BINARY` | 1.00 | 1 bit/weight | {-1, +1} |
| `TERNARY` | 1.58 | 2 bits/weight | {-1, 0, +1} |
| `OIL2` | 2.00 | 2 bits/weight | 4 centroids |
| `OIL4` | 1.50 | 4 bits/weight + 16×FP16 table | 16 centroids |
| `OIL8` | 0.85 | 8 bits/weight + 256×FP32 table | 256 centroids |
| `OIL16` | 16.0 | 16 bits/weight | — |
| `OIL32` | 32.0 | 32 bits/weight | FP32 lossless |

## Quantization Format Details

### OIL4
```
Centroid table: 16 × FP16 = 32 bytes
Index array: 4 bits per weight (2 weights per byte)
Effective BPW: (32 + n*0.5) / n ≈ 0.5 (for large n in bytes per weight)
              + centroid per-weight overhead ≈ 1.50 BPW total
```

### OIL8
```
Centroid table: 256 × FP32 = 1024 bytes
Index array: 8 bits per weight (1 weight per byte)
Effective BPW: 8 + (1024 * 8 / n) ≈ 0.85 (for large n)
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
