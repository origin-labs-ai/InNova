// kernel_q12.cpp — SIMD GEMM kernel for Q12 (12-bit, 4096 centroids)
// 12-bit packed indices: 2 weights per 3 bytes (24 bits = 2 × 12 bits)
// Codebook: 4096 × FP16 centroids (stored as uint16_t, converted to FP32 at runtime)
#include "quant/kernel.h"
#include "quant/tensor.h"
#include "quant/codebook.h"
#include <cstring>
#include <cmath>
#include <cstdint>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace quant {
namespace kernel {

// FP16 bits -> float (same as kernel_quant4.cpp)
static float q12_fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        float v = (float)mant * 0.000000059604644775390625f;
        return sign ? -v : v;
    } else if (exp == 31) {
        return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

// Unpack 2 × 12-bit indices from 3 bytes
// [b0][b1][b2]: idx0 = bits[11:0], idx1 = bits[23:12]
static inline void unpack_q12_2(const uint8_t* src, uint16_t* dst) {
    uint32_t bits = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
    dst[0] = (uint16_t)(bits & 0xFFF);
    dst[1] = (uint16_t)((bits >> 12) & 0xFFF);
}

[[maybe_unused]] static void q12_gemm_scalar(const uint8_t* packed_indices, const uint16_t* codebook_fp16,
                              const float* activations, float* output,
                              int M, int N, int K) {
    // Pre-convert codebook to FP32 for the hot loop
    // 4096 centroids = 16KB FP32 cache — fits in L1
    float codebook[4096];
    for (int i = 0; i < 4096; i++) {
        codebook[i] = q12_fp16_to_float(codebook_fp16[i]);
    }

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            int packed_row_bytes = (K * 12 + 7) / 8; // 1.5 bytes per weight
            const uint8_t* w_row = packed_indices + ((int64_t)m * N + n) * packed_row_bytes;
            const float* a_row = activations + (int64_t)n * K;

#if defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(w_row + packed_row_bytes), _MM_HINT_T0);
#else
            __builtin_prefetch(w_row + packed_row_bytes, 0, 3);
#endif

            int k = 0;
            // Process 2 weights at a time (3 bytes = 2 × 12-bit indices)
            for (; k + 2 <= K; k += 2) {
                uint16_t indices[2];
                unpack_q12_2(w_row + (k * 3) / 2, indices);
                sum += codebook[indices[0]] * a_row[k];
                sum += codebook[indices[1]] * a_row[k + 1];
            }
            // Handle last weight if K is odd
            if (k < K) {
                uint16_t indices[2] = {};
                uint8_t temp[3] = {};
                temp[0] = w_row[(k * 3) / 2];
                if ((k * 3) / 2 + 1 < packed_row_bytes) temp[1] = w_row[(k * 3) / 2 + 1];
                unpack_q12_2(temp, indices);
                sum += codebook[indices[0]] * a_row[k];
            }
            output[m * N + n] = sum;
        }
    }
}

#if defined(__AVX2__)
static void q12_gemm_avx2(const uint8_t* packed_indices, const uint16_t* codebook_fp16,
                           const float* activations, float* output,
                           int M, int N, int K) {
    // Pre-convert codebook to FP32
    float codebook[4096];
    for (int i = 0; i < 4096; i++) {
        codebook[i] = q12_fp16_to_float(codebook_fp16[i]);
    }

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m256 sum8 = _mm256_setzero_ps();
            int packed_row_bytes = (K * 12 + 7) / 8;
            const uint8_t* w_row = packed_indices + ((int64_t)m * N + n) * packed_row_bytes;
            const float* a_row = activations + (int64_t)n * K;

            int k = 0;
            // Process 8 weights at a time: 4 × (2 × 12-bit) = 12 bytes
            for (; k + 8 <= K; k += 8) {
                int byte_offset = (k * 3) / 2;
                uint16_t idx_buf[8];
                // Unpack 4 pairs of 12-bit indices
                unpack_q12_2(w_row + byte_offset, idx_buf);
                unpack_q12_2(w_row + byte_offset + 3, idx_buf + 2);
                unpack_q12_2(w_row + byte_offset + 6, idx_buf + 4);
                unpack_q12_2(w_row + byte_offset + 9, idx_buf + 6);

#if defined(_MSC_VER)
                _mm_prefetch(reinterpret_cast<const char*>(w_row + byte_offset + 12), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(a_row + k + 8), _MM_HINT_T0);
#else
                __builtin_prefetch(w_row + byte_offset + 12, 0, 3);
                __builtin_prefetch(a_row + k + 8, 0, 3);
#endif

                // Gather 8 codebook entries
                __m256i idx = _mm256_set_epi32(
                    (int)idx_buf[7], (int)idx_buf[6], (int)idx_buf[5], (int)idx_buf[4],
                    (int)idx_buf[3], (int)idx_buf[2], (int)idx_buf[1], (int)idx_buf[0]);
                __m256 w = _mm256_i32gather_ps(codebook, idx, 4);
                __m256 a = _mm256_loadu_ps(a_row + k);
                sum8 = _mm256_fmadd_ps(w, a, sum8);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum8, 1);
            __m128 lo = _mm256_castps256_ps128(sum8);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float result = _mm_cvtss_f32(sum4);

            // Handle tail
            for (; k + 2 <= K; k += 2) {
                uint16_t indices[2];
                unpack_q12_2(w_row + (k * 3) / 2, indices);
                result += codebook[indices[0]] * a_row[k];
                result += codebook[indices[1]] * a_row[k + 1];
            }
            if (k < K) {
                uint16_t indices[2] = {};
                uint8_t temp[3] = {};
                temp[0] = w_row[(k * 3) / 2];
                if ((k * 3) / 2 + 1 < packed_row_bytes) temp[1] = w_row[(k * 3) / 2 + 1];
                unpack_q12_2(temp, indices);
                result += codebook[indices[0]] * a_row[k];
            }
            output[m * N + n] = result;
        }
    }
}
#endif

void q12_gemm(const uint8_t* packed_indices, const uint16_t* codebook_fp16,
              const float* activations, float* output,
              int M, int N, int K) {
#if defined(__AVX2__)
    q12_gemm_avx2(packed_indices, codebook_fp16, activations, output, M, N, K);
#else
    q12_gemm_scalar(packed_indices, codebook_fp16, activations, output, M, N, K);
#endif
}

} // namespace kernel
} // namespace quant
