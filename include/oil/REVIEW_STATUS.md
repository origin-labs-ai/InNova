# REVIEW_STATUS.md — MYTHOS.cpp Per-File Status Matrix

**Last updated:** 2026-07-26

## Status Legend

- **Y** = Pass — no issues in this category
- **N** = Fail — issues found (see PEER_REVIEW_LOG.md for details)
- **~** = Partial — minor issues present but not blocking

## Per-File Status Matrix

| File | Reviewed | Clean Build | Thread Safe | No Stubs | No UB | No Magic Numbers | Overall |
|------|----------|-------------|-------------|----------|-------|-------------------|---------|
| `src/format_registry.cpp` | Y | Y | Y | Y | ~ (k=1 div-by-zero) | N (1e-8, 0.51, 1e30) | ~ MINOR |
| `src/native_trainer.cpp` | Y | ~ (dup include) | Y | Y | N (const-cast, always-0 metric) | N (1e-12, 1e30) | N NEEDS_WORK |
| `include/oil/kernel_production.h` | Y | Y | Y | Y | Y | Y | Y CLEAN |
| `src/kernel_production.cpp` | Y | Y | Y | ~ (dead vars) | N (AVX2 double-count w4*a0) | N (3GHz assumption) | N NEEDS_WORK |
| `src/format_planner.cpp` | Y | Y | Y | Y | Y | N (regime_switch=2.25) | ~ MINOR |
| `include/oil/format_planner.h` | Y | Y | Y | N (HESSIAN_DIAG unused) | Y | Y | ~ MINOR |
| `src/transformer.cpp` | Y | Y | Y | Y | N (AVX2 OOB <8, strided load) | N (seed=42) | N NEEDS_WORK |
| `include/oil/transformer.h` | Y | Y | Y | ~ (use_parallel_residual) | Y | Y | ~ MINOR |
| `src/flash_attention.cpp` | Y | Y | Y | ~ (dropout_p unused, cfg_ unused) | Y | N (block=32/16) | ~ MINOR |
| `include/oil/flash_attention.h` | Y | Y | Y | N (online_softmax_tile undeclared) | Y | Y | ~ MINOR |
| `src/kv_cache.cpp` | Y | Y | N (const_cast UB) | Y | ~ (infinite loop edge case) | N (FP8_MAX=127) | N NEEDS_WORK |
| `src/autograd_engine.cpp` | Y | Y | Y | Y | ~ (raw ptr lifetime) | Y | ~ MINOR |

## Detailed Status by Category

### Clean Build
All files include required headers and have no syntax errors visible at the source level. One duplicate `#include <algorithm>` in `native_trainer.cpp`.

### Thread Safety
- **AutogradEngine**: Uses `std::mutex` and `std::atomic<bool>` correctly. Safe for single-instance concurrent access.
- **KVCache (paged variants)**: NOT thread-safe. `mutable` members and `const` methods that mutate state. `const_cast` in `load_from_disk` is UB if object is const-declared.
- **FormatRegistry**: All static methods; data is in C++11 thread-safe static locals. Safe.
- **NativeOILTrainer**: Not designed for concurrent use. Acceptable.
- **FormatPlanner**: Single-threaded planner. Acceptable.

### No Stubs
- `online_softmax_tile` in `flash_attention.h:26` — declared, never defined
- `HESSIAN_DIAG` in `format_planner.h:39` — enum value, never implemented
- `dropout_p` parameter in `flash_attention_forward` — accepted, never applied
- `FlashAttentionConfig::block_size`, `softmax_scale` — stored in cfg_, never read

### No Undefined Behavior
- **kernel_production.cpp:256** — Critical: AVX2 OIL8 GEMV double-counts w4*a0 product
- **transformer.cpp:280** — Critical: Negative array index when S_full < 8 under AVX2
- **transformer.cpp:354-367** — Critical: Strided memory load treated as contiguous in AVX2 attention output
- **kv_cache.cpp:412** — Major: `const_cast` to mutate object that may be const-declared
- **format_registry.cpp:140** — Major: Division by zero when k=1 in Lloyd-Max
- **autograd_engine.cpp:351** — Minor: `const_cast<void*>` on tensor data pointer

### No Magic Numbers
Magic numbers identified across the codebase:
- `1e-8f` (Lloyd-Max convergence), `1e-12f` (scale floors), `1e-30f` (sentinel), `1e-10f` (scale floor)
- `0.51f` (BPW tolerance), `0.5f` (ternary threshold)
- `2.25f` (regime switch), `3.0e9` (assumed clock speed)
- `32` (OIL_BLOCK_SIZE), `256` (quantize block_size), `42` (RNG seed)
- `0x4F494C4E` (checkpoint magic number)
- `127.0f` (FP8_MAX), `64` (FP8_BLOCK_SIZE)

Most are standard numerical engineering constants. A few should be named (`0.51f`, `2.25f`, `3.0e9`).

## Critical Action Items

| Priority | File | Line | Issue |
|----------|------|------|-------|
| P0 | `kernel_production.cpp` | 256 | AVX2 OIL8 GEMV double-counts w4*a0 — all quantized inference wrong on AVX2 |
| P0 | `transformer.cpp` | 280 | AVX2 softmax OOB read when S_full < 8 |
| P0 | `transformer.cpp` | 354 | AVX2 attention output loads v from wrong memory layout |
| P0 | `native_trainer.cpp` | 201 | frozen_fraction always 0 (integer division) |
| P1 | `kv_cache.cpp` | 411 | Infinite loop when physical_memory_limit_ = 0 |
| P1 | `format_registry.cpp` | 140 | Lloyd-Max div-by-zero when k=1 |
| P1 | `kv_cache.cpp` | 412 | const_cast UB in const method |

---

*Status matrix generated: 2026-07-26*
