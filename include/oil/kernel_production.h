// ============================================================================
// kernel_production.h — Production SIMD kernels for OIL inference
// ----------------------------------------------------------------------------
// Tiled GEMV, batch GEMV, and calibration-aware kernels for:
//   SPARK_Q0 (1.50-bit, 4 values/byte), OIL4 (4-bit, 2/byte), OIL8 (8-bit, 1/byte)
//
// ISA dispatch: AVX2 > SSE4.1 > Scalar (runtime CPU detection)
// ============================================================================
#pragma once
#include "oil/types.h"
#include <cstdint>
#include <cstddef>

namespace oil {
namespace adapters {
namespace prod {

// ── ISA detection ──────────────────────────────────────────────────────────
int detect_cpu_isa();  // 2=AVX2, 1=SSE4.1, 0=scalar

// ── Tiled GEMV (cache-blocked M dimension) ─────────────────────────────────
// y = scale * (W @ x), where W is in OIL format.
// tile_m = rows processed per cache-blocking step (default: L1/line_size)

void gemv_spark_tiled(const uint8_t* packed_w, float scale,
                      const float* x, float* y,
                      int M, int K, int tile_m = 32);

void gemv_oil4_tiled(const uint8_t* packed_indices, const uint16_t* codebook,
                     float scale, const float* x, float* y,
                     int M, int K, int tile_m = 32);

void gemv_oil8_tiled(const uint8_t* indices, const float* codebook,
                     float scale, const float* x, float* y,
                     int M, int K, int tile_m = 32);

// ── Batch GEMV (multiple activation vectors) ──────────────────────────────
// Y[m, n] = scale * (W[m, :] @ X[n, :])
// M = output rows, N = batch size, K = inner dimension

void gemv_spark_batch(const uint8_t* packed_w, float scale,
                      const float* X, float* Y,
                      int M, int N, int K);

void gemv_oil4_batch(const uint8_t* packed_indices, const uint16_t* codebook,
                     float scale, const float* X, float* Y,
                     int M, int N, int K);

void gemv_oil8_batch(const uint8_t* indices, const float* codebook,
                     float scale, const float* X, float* Y,
                     int M, int N, int K);

// ── Calibration-aware importance scoring ──────────────────────────────────
// Given weights + sample activations, compute per-block importance scores.
// Higher score → more important → deserves OIL8 or OIL4.
// importance array must be pre-allocated with num_blocks elements.

void calibrate_spark_importance(const float* weights, const float* activations,
                                float* importance, int M, int K, int block_size);

void calibrate_oil4_importance(const float* weights, const float* activations,
                               float* importance, int M, int K, int block_size);

void calibrate_oil8_importance(const float* weights, const float* activations,
                               float* importance, int M, int K, int block_size);

// ── Performance profiling hooks ────────────────────────────────────────────
struct KernelPerf {
    uint64_t cycles = 0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    double   gflops = 0.0;
};

KernelPerf profile_gemv_spark(const uint8_t* packed_w, float scale,
                              const float* x, float* y, int M, int K, int repeats = 100);

KernelPerf profile_gemv_oil4(const uint8_t* packed_indices, const uint16_t* codebook,
                             float scale, const float* x, float* y, int M, int K, int repeats = 100);

KernelPerf profile_gemv_oil8(const uint8_t* indices, const float* codebook,
                             float scale, const float* x, float* y, int M, int K, int repeats = 100);

} // namespace prod
} // namespace adapters
} // namespace oil
