#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <cstdint>

namespace oil {
namespace kernel {

// TL1: SPARK Lookup Table, groups of 2
void tl1_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K);
void tl1_precompute_lut(const int8_t* activations, int8_t* lut,
                        int K, float scales);

// TL2: SPARK Lookup Table, groups of 3 (element-wise mirror consolidation)
void tl2_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K);
void tl2_precompute_lut(const int8_t* activations, int8_t* lut,
                        int K, float scales);

// OIL8: Codebook lookup GEMM
// weights = uint8_t indices, codebook = float[256], activations = float
void oil8_gemm(const uint8_t* indices, const float* codebook,
               const float* activations, float* output,
               int M, int N, int K);

// OIL4: Codebook lookup GEMM
// weights = nibble packed indices, codebook = uint16_t[16] (FP16), activations = float
void oil4_gemm(const uint8_t* packed_indices, const uint16_t* codebook,
               const float* activations, float* output,
               int M, int N, int K);

// Scalar fallback matmul
void scalar_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K);

// Tiled GEMM with 64x64 blocking for cache efficiency
void tiled_gemm(const float* A, const float* B, float* C,
                int M, int N, int K);

// AVX2 matmul
void avx2_gemm(const float* A, const float* B, float* C,
               int M, int N, int K);

// AVX2 tiled GEMM
void avx2_tiled_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K);

} // namespace kernel
} // namespace oil
