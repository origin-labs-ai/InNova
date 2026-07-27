#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <mutex>

namespace oil {
namespace kernel {

// I2_S unpack: extract SPARK values {-1,0,+1} from packed 2-bit
// Storage: 4 values per byte: [v3:2][v2:2][v1:2][v0:2]
// Mapping: 00=-1, 01=0, 10=+1
static inline int decode_i2s(uint8_t packed, int shift) {
    int val = (packed >> shift) & 3;
    if (val == 0) return -1;
    if (val == 1) return 0;
    return 1;
}

// Pre-computed LUT: for each byte value (0-255), store all 4 decoded SPARK values
static int8_t g_i2s_lut[256][4];
static std::once_flag g_i2s_lut_init;
static void init_i2s_lut() {
    for (int i = 0; i < 256; i++)
        for (int s = 0; s < 4; s++)
            g_i2s_lut[i][s] = (int8_t)decode_i2s((uint8_t)i, s * 2);
}

void i2s_gemv(const uint8_t* packed_w, float w_scale,
              const int8_t* act, float act_scale,
              float* output, int K) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += (float)g_i2s_lut[packed_w[i / 4]][i % 4] * (float)act[i];
    }
    *output = sum * w_scale * act_scale;
}

void i2s_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K) {
    std::call_once(g_i2s_lut_init, init_i2s_lut);
    const uint8_t* w = (const uint8_t*)weights.data();
    const float* a = (const float*)activations.data();
    float* o = (float*)output.data();
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k += 128) {
                int block_k = (K - k < 128) ? K - k : 128;
                const uint8_t* wb = w + m * (K / 4) + k / 4;
                const float* ab = a + n * K + k;
                float block_sum = 0;
                for (int i = 0; i < block_k; i++) {
                    block_sum += (float)g_i2s_lut[wb[i / 4]][i % 4] * ab[i];
                }
                sum += block_sum;
            }
            o[m * N + n] = sum;
        }
    }
}

void tiled_gemm(const float* A, const float* B, float* C,
                int M, int N, int K) {
    const int TILE = 64;
    std::fill(C, C + M * N, 0.0f);

    for (int i = 0; i < M; i += TILE) {
        int imax = (std::min)(i + TILE, M);
        for (int j = 0; j < N; j += TILE) {
            int jmax = (std::min)(j + TILE, N);
            for (int k = 0; k < K; k += TILE) {
                int kmax = (std::min)(k + TILE, K);
                for (int ti = i; ti < imax; ti++) {
                    for (int tk = k; tk < kmax; tk++) {
                        float a_val = A[ti * K + tk];
                        for (int tj = j; tj < jmax; tj++) {
                            C[ti * N + tj] += a_val * B[tk * N + tj];
                        }
                    }
                }
            }
        }
    }
}

void scalar_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

#if defined(OIL_AVX2)
#include <immintrin.h>

void avx2_gemm(const float* A, const float* B, float* C,
               int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n += 8) {
            __m256 cvec = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 avec = _mm256_set1_ps(A[m * K + k]);
                __m256 bvec = _mm256_loadu_ps(&B[k * N + n]);
                cvec = _mm256_fmadd_ps(avec, bvec, cvec);
            }
            _mm256_storeu_ps(&C[m * N + n], cvec);
        }
    }
}

void avx2_tiled_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 256;
    std::fill(C, C + M * N, 0.0f);

    for (int i = 0; i < M; i += TILE_M) {
        int imax = (std::min)(i + TILE_M, M);
        for (int j = 0; j < N; j += TILE_N) {
            int jmax = (std::min)(j + TILE_N, N);
            for (int k = 0; k < K; k += TILE_K) {
                int kmax = (std::min)(k + TILE_K, K);
                for (int ti = i; ti < imax; ti++) {
                    for (int tk = k; tk < kmax; tk++) {
                        __m256 avec = _mm256_set1_ps(A[ti * K + tk]);
                        int tj;
                        for (tj = j; tj + 8 <= jmax; tj += 8) {
                            __m256 bvec = _mm256_loadu_ps(&B[tk * N + tj]);
                            __m256 cvec = _mm256_loadu_ps(&C[ti * N + tj]);
                            cvec = _mm256_fmadd_ps(avec, bvec, cvec);
                            _mm256_storeu_ps(&C[ti * N + tj], cvec);
                        }
                        for (; tj < jmax; tj++) {
                            C[ti * N + tj] += A[ti * K + tk] * B[tk * N + tj];
                        }
                    }
                }
            }
        }
    }
}

// AVX2 I2S GEMM with LUT-based decode (processes 32 SPARK values per iteration)
void i2s_gemm_avx2(const Tensor& weights, const Tensor& activations,
                   Tensor& output, int M, int N, int K) {
    std::call_once(g_i2s_lut_init, init_i2s_lut);
    const uint8_t* w = (const uint8_t*)weights.data();
    const float* a = (const float*)activations.data();
    float* o = (float*)output.data();

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m256 sum = _mm256_setzero_ps();
            int k;
            for (k = 0; k + 32 <= K; k += 32) {
                const uint8_t* wb = w + m * (K / 4) + k / 4;
                const float* ab = a + n * K + k;
                // Load 8 packed bytes = 32 SPARK values, decode into 32 floats
                int32_t dec[32];
                for (int i = 0; i < 8; i++) {
                    uint8_t b = wb[i];
                    dec[i * 4 + 0] = g_i2s_lut[b][0];
                    dec[i * 4 + 1] = g_i2s_lut[b][1];
                    dec[i * 4 + 2] = g_i2s_lut[b][2];
                    dec[i * 4 + 3] = g_i2s_lut[b][3];
                }
                // Accumulate 4 x 8-way FMAs
                for (int j = 0; j < 4; j++) {
                    __m256 wv = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)&dec[j * 8]));
                    __m256 av = _mm256_loadu_ps(&ab[j * 8]);
                    sum = _mm256_fmadd_ps(wv, av, sum);
                }
            }
            // Remainder
            float tail_sum = 0;
            for (; k < K; k++) {
                const uint8_t* wb = w + m * (K / 4) + k / 4;
                tail_sum += (float)g_i2s_lut[wb[0]][k % 4] * a[n * K + k];
            }
            float hsum[8];
            _mm256_storeu_ps(hsum, sum);
            float total = hsum[0] + hsum[1] + hsum[2] + hsum[3]
                        + hsum[4] + hsum[5] + hsum[6] + hsum[7] + tail_sum;
            o[m * N + n] = total;
        }
    }
}

void i2s_gemm_vnni(const uint8_t* packed_w, const int8_t* activations,
                   float* output, int M, int N, int K) {
    std::call_once(g_i2s_lut_init, init_i2s_lut);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m256 sum = _mm256_setzero_ps();
            const uint8_t* wb = packed_w + m * (K / 4);
            const int8_t* ab = activations + n * K;
            int k;
            for (k = 0; k + 32 <= K; k += 32) {
                int32_t dec[32];
                for (int i = 0; i < 8; i++) {
                    uint8_t b = wb[(k + i * 4) / 4];
                    dec[i * 4 + 0] = g_i2s_lut[b][0];
                    dec[i * 4 + 1] = g_i2s_lut[b][1];
                    dec[i * 4 + 2] = g_i2s_lut[b][2];
                    dec[i * 4 + 3] = g_i2s_lut[b][3];
                }
                for (int j = 0; j < 4; j++) {
                    __m256 wv = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)&dec[j * 8]));
                    __m128i ai8 = _mm_loadu_si128((__m128i*)&ab[k + j * 8]);
                    __m256 av = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ai8));
                    sum = _mm256_fmadd_ps(wv, av, sum);
                }
            }
            float hsum[8];
            _mm256_storeu_ps(hsum, sum);
            float total = hsum[0] + hsum[1] + hsum[2] + hsum[3]
                        + hsum[4] + hsum[5] + hsum[6] + hsum[7];
            for (; k < K; k++)
                total += (float)g_i2s_lut[wb[k / 4]][k % 4] * (float)ab[k];
            output[m * N + n] = total;
        }
    }
}

#else
void avx2_gemm(const float* A, const float* B, float* C,
               int M, int N, int K) {
    scalar_gemm(A, B, C, M, N, K);
}

void avx2_tiled_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    tiled_gemm(A, B, C, M, N, K);
}

void i2s_gemm_avx2(const Tensor& weights, const Tensor& activations,
                   Tensor& output, int M, int N, int K) {
    i2s_gemm(weights, activations, output, M, N, K);
}

void i2s_gemm_vnni(const uint8_t* packed_w, const int8_t* activations,
                   float* output, int M, int N, int K) {
    std::call_once(g_i2s_lut_init, init_i2s_lut);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += (float)g_i2s_lut[packed_w[m * (K / 4) + k / 4]][k % 4]
                     * (float)activations[n * K + k];
            }
            output[m * N + n] = sum;
        }
    }
}
#endif

} // namespace kernel
} // namespace oil
