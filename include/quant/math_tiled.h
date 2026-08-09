#pragma once

#include "quant/tensor.h"
#include <cstdint>

namespace quant {
namespace math {

// ---------------------------------------------------------------------------
// Tile sizes tuned for L1 (32 KB) / L2 (256 KB) cache.
// 64×64 FP32 tile = 64×64×4 = 16 KB fits in L1.
// ---------------------------------------------------------------------------
constexpr int64_t TILE_M = 64;
constexpr int64_t TILE_N = 64;
constexpr int64_t TILE_K = 64;

enum class Transpose { None, TransA, TransB, TransAB };

using quant::Activation;

// ---------------------------------------------------------------------------
// C = alpha * op(A) * op(B) + beta * C
// Row-major (default) or column-major via the `col_major` flag.
// ---------------------------------------------------------------------------
void gemm_tiled(float alpha,
                const float* A, int64_t lda,
                const float* B, int64_t ldb,
                float beta,
                float* C, int64_t ldc,
                int64_t M, int64_t N, int64_t K,
                Transpose trans = Transpose::None,
                bool col_major = false);

// ---------------------------------------------------------------------------
// Fused bias: C[i][j] += bias[j] after GEMM
// ---------------------------------------------------------------------------
void gemm_tiled_bias(float alpha,
                     const float* A, int64_t lda,
                     const float* B, int64_t ldb,
                     float beta,
                     float* C, int64_t ldc,
                     int64_t M, int64_t N, int64_t K,
                     const float* bias,
                     Transpose trans = Transpose::None);

// ---------------------------------------------------------------------------
// Fused GEMM + activation: C = act(alpha * op(A) * op(B) + beta * C)
// ---------------------------------------------------------------------------
void gemm_tiled_act(float alpha,
                    const float* A, int64_t lda,
                    const float* B, int64_t ldb,
                    float beta,
                    float* C, int64_t ldc,
                    int64_t M, int64_t N, int64_t K,
                    Activation act = Activation::ReLU,
                    Transpose trans = Transpose::None);

// ---------------------------------------------------------------------------
// Fused GEMM + bias + activation
// ---------------------------------------------------------------------------
void gemm_tiled_bias_act(float alpha,
                         const float* A, int64_t lda,
                         const float* B, int64_t ldb,
                         float beta,
                         float* C, int64_t ldc,
                         int64_t M, int64_t N, int64_t K,
                         const float* bias,
                         Activation act = Activation::ReLU,
                         Transpose trans = Transpose::None);

// ---------------------------------------------------------------------------
// Tensor-level convenience wrappers (row-major, contiguous).
// ---------------------------------------------------------------------------
void gemm_tiled(float alpha, const Tensor& A, const Tensor& B,
                float beta, Tensor& C,
                Transpose trans = Transpose::None);

void gemm_tiled_bias(float alpha, const Tensor& A, const Tensor& B,
                     float beta, Tensor& C, const Tensor& bias,
                     Transpose trans = Transpose::None);

void gemm_tiled_act(float alpha, const Tensor& A, const Tensor& B,
                    float beta, Tensor& C,
                    Activation act = Activation::ReLU,
                    Transpose trans = Transpose::None);

void gemm_tiled_bias_act(float alpha, const Tensor& A, const Tensor& B,
                         float beta, Tensor& C, const Tensor& bias,
                         Activation act = Activation::ReLU,
                         Transpose trans = Transpose::None);

} // namespace math
} // namespace quant
