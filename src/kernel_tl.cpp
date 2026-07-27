#include "oil/kernel.h"
#include "oil/tensor.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace oil {
namespace kernel {

// TL1 precompute: group of 2 SPARK weights
// 9 possible sums {-2,-1,0,1,2} * scales
void tl1_precompute_lut(const int8_t* activations, int8_t* lut,
                        int K, [[maybe_unused]] float scales) {
    int groups = K / 2;
    [[maybe_unused]] int remaining = K % 2;
    for (int k = 0; k < groups; k++) {
        int8_t a0 = activations[k * 2];
        int8_t a1 = activations[k * 2 + 1];
        int idx = 0;
        for (int w0 = -1; w0 <= 1; w0++) {
            for (int w1 = -1; w1 <= 1; w1++) {
                int32_t sum = (int32_t)a0 * w0 + (int32_t)a1 * w1;
                lut[idx] = (int8_t)std::clamp(sum, -128, 127);
                idx++;
            }
        }
        lut += 9;
    }
}

void tl1_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K) {
    const uint8_t* w = (const uint8_t*)weights.data();
    const float* a = (const float*)activations.data();

    Tensor a_int8(Shape{N * K}, DType::U8);
    int8_t* a_i8 = (int8_t*)a_int8.data();

    std::vector<float> scales(N);
    for (int n = 0; n < N; n++) {
        float max_abs = 0;
        for (int k = 0; k < K; k++) {
            float abs_v = std::fabs(a[n * K + k]);
            if (abs_v > max_abs) max_abs = abs_v;
        }
        if (max_abs < 1e-10f) max_abs = 1e-10f;
        scales[n] = 127.0f / max_abs;
        for (int k = 0; k < K; k++)
            a_i8[n * K + k] = (int8_t)std::round(a[n * K + k] * scales[n]);
    }

    // Build LUT and compute
    int total_lut_groups = K / 2;
    std::vector<int8_t> lut(total_lut_groups * 9);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            const uint8_t* w_row = w + (m * N + n) * K / 4;
            const int8_t* a_row = a_i8 + n * K;
            int32_t total = 0;

            tl1_precompute_lut(a_row, lut.data(), K, 1.0f);

            for (int i = 0; i < K - 1; i += 2) {
                uint8_t packed = w_row[i / 4];
                int wi = (packed >> ((i % 4) * 2)) & 3;
                total += lut[(i / 2) * 9 + wi];
            }

            ((float*)output.data())[m * N + n] = (float)total / scales[n];
        }
    }
}

// TL2: Group of 3 SPARK weights → 27 entry LUT per group
// 3 SPARK weights per group, each {-1,0,+1} → 3^3 = 27 combinations
// Storage: 4 groups of 3 weights packed into 3 bytes (12 bits for 4×3=12 values)
// We use the same SPARK encoding: 00=-1, 01=0, 10=+1 (2 bits each)

static inline int decode_tl2(uint8_t packed, int shift) {
    int val = (packed >> shift) & 3;
    if (val == 0) return -1;
    if (val == 1) return 0;
    return 1;
}

void tl2_precompute_lut(const int8_t* activations, int8_t* lut,
                        int K, [[maybe_unused]] float scales) {
    for (int k = 0; k < K; k += 3) {
        int8_t a0 = activations[k];
        int8_t a1 = activations[k + 1];
        int8_t a2 = activations[k + 2];
        int idx = 0;
        for (int w0 = -1; w0 <= 1; w0++) {
            for (int w1 = -1; w1 <= 1; w1++) {
                for (int w2 = -1; w2 <= 1; w2++) {
                    int32_t sum = (int32_t)a0 * w0 + (int32_t)a1 * w1 + (int32_t)a2 * w2;
                    lut[idx] = (int8_t)std::clamp(sum, -128, 127);
                    idx++;
                }
            }
        }
        lut += 27;
    }
}

void tl2_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K) {
    const uint8_t* w = (const uint8_t*)weights.data();
    const float* a = (const float*)activations.data();

    Tensor a_int8(Shape{N * K}, DType::U8);
    int8_t* a_i8 = (int8_t*)a_int8.data();

    // Quantize activations per row
    std::vector<float> scales(N);
    for (int n = 0; n < N; n++) {
        float max_abs = 0;
        for (int k = 0; k < K; k++) {
            float abs_v = std::fabs(a[n * K + k]);
            if (abs_v > max_abs) max_abs = abs_v;
        }
        if (max_abs < 1e-10f) max_abs = 1e-10f;
        scales[n] = 127.0f / max_abs;
        for (int k = 0; k < K; k++)
            a_i8[n * K + k] = (int8_t)std::round(a[n * K + k] * scales[n]);
    }

    // TL2 packing: 12 weights in 3 bytes, 4 groups of 3
    int bytes_per_row = (K * 2 + 7) / 8; // 2 bits per weight

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            const uint8_t* w_row = w + (m * N + n) * bytes_per_row;
            const int8_t* a_row = a_i8 + n * K;
            int64_t total = 0;

            for (int g = 0; g * 12 < K; g++) {
                int rem = K - g * 12;
                int n_triples = std::min(4, (rem + 2) / 3);
                if (n_triples <= 0) break;

                int vals[12];
                for (int b = 0; b < 3; b++) {
                    uint8_t byte = w_row[g * 3 + b];
                    vals[b * 4 + 0] = decode_tl2(byte, 0);
                    vals[b * 4 + 1] = decode_tl2(byte, 2);
                    vals[b * 4 + 2] = decode_tl2(byte, 4);
                    vals[b * 4 + 3] = decode_tl2(byte, 6);
                }

                for (int t = 0; t < n_triples; t++) {
                    int base = g * 12 + t * 3;
                    int w0 = vals[t * 3 + 0];
                    int w1 = vals[t * 3 + 1];
                    int w2 = vals[t * 3 + 2];
                    total += (int)a_row[base] * w0 +
                             (int)a_row[base + 1] * w1 +
                             (int)a_row[base + 2] * w2;
                }
            }

            ((float*)output.data())[m * N + n] = (float)total / scales[n];
        }
    }
}
} // namespace kernel
} // namespace oil
