#include "oil/oil_engines.h"
#include "oil/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace oil {
namespace engines {

// ===========================================================================
// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa bits
// Range: [-448, 448], no inf/nan (uses all 8-bit patterns for values)
// ===========================================================================

float fp8_e4m3_dequantize(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp_raw = (bits >> 3) & 0xF;
    int mant = bits & 0x7;
    float val;
    if (exp_raw == 0) {
        val = (float)mant * (1.0f / 64.0f);
    } else if (exp_raw == 0xF && mant == 0x7) {
        val = 0.0f;
    } else {
        int exp = exp_raw - 7;
        val = ldexpf(1.0f + (float)mant / 8.0f, exp);
    }
    return sign ? -val : val;
}

uint8_t fp8_e4m3_quantize(float val) {
    if (val == 0.0f) return 0;
    int sign = 0;
    if (val < 0) { sign = 1; val = -val; }
    if (val > 448.0f) val = 448.0f;
    int exp = 0;
    float mant = val;
    while (mant >= 2.0f && exp < 8) { mant /= 2.0f; exp++; }
    while (mant < 1.0f && exp > -7) { mant *= 2.0f; exp--; }
    if (exp < -7) {
        int m = (int)std::round(mant * 64.0f);
        if (m > 7) m = 7;
        return (uint8_t)((sign << 7) | m);
    }
    int m = (int)std::round((mant - 1.0f) * 8.0f);
    if (m > 7) m = 7;
    if (m < 0) m = 0;
    return (uint8_t)((sign << 7) | ((exp + 7) << 3) | m);
}

Tensor fp8_e4m3_dequant_tensor(const uint8_t* data, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
#ifdef OIL_HAS_AVX2
    if (n >= 16) {
        dequant_tensor_fp8_avx2(data, od, n, false);
        return out;
    }
#endif
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e4m3_dequantize(data[i]);
    return out;
}

// ===========================================================================
// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa bits
// Range: [-57344, 57344], supports inf/nan
// ===========================================================================

float fp8_e5m2_dequantize(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp_raw = (bits >> 2) & 0x1F;
    int mant = bits & 0x3;
    float val;
    if (exp_raw == 0) {
        val = (float)mant * (1.0f / 16384.0f);
    } else if (exp_raw == 0x1F) {
        val = (mant == 0) ? INFINITY : NAN;
    } else {
        int exp = exp_raw - 15;
        val = ldexpf(1.0f + (float)mant / 4.0f, exp);
    }
    return sign ? -val : val;
}

uint8_t fp8_e5m2_quantize(float val) {
    if (val == 0.0f) return 0;
    int sign = 0;
    if (val < 0) { sign = 1; val = -val; }
    if (val > 57344.0f) val = 57344.0f;
    int exp = 0;
    float mant = val;
    while (mant >= 2.0f && exp < 16) { mant /= 2.0f; exp++; }
    while (mant < 1.0f && exp > -15) { mant *= 2.0f; exp--; }
    if (exp < -15) {
        int m = (int)std::round(mant * 16384.0f);
        if (m > 3) m = 3;
        return (uint8_t)((sign << 7) | m);
    }
    int m = (int)std::round((mant - 1.0f) * 4.0f);
    if (m > 3) m = 3;
    if (m < 0) m = 0;
    return (uint8_t)((sign << 7) | ((exp + 15) << 2) | m);
}

Tensor fp8_e5m2_dequant_tensor(const uint8_t* data, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
#ifdef OIL_HAS_AVX2
    if (n >= 16) {
        dequant_tensor_fp8_avx2(data, od, n, true);
        return out;
    }
#endif
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e5m2_dequantize(data[i]);
    return out;
}

// ===========================================================================
// NF4: Normal Float 4-bit quantization format
// 16 values from normal distribution, quantized to 4 bits
// ===========================================================================

static const float NF4_CODEBOOK[16] = {
    -1.0f, -0.6961928f, -0.525073f, -0.39497948f,
    -0.28444138f, -0.18477343f, -0.09105138f, 0.0f,
    0.07958097f, 0.16092212f, 0.24611243f, 0.33712566f,
    0.43570983f, 0.54554284f, 0.67132018f, 1.0f
};

uint8_t nf4_quantize(float val, float scale) {
    float normalized = val / (scale + 1e-10f);
    normalized = std::max(-1.0f, std::min(1.0f, normalized));
    int best = 0;
    float best_dist = 1e10f;
    for (int i = 0; i < 16; ++i) {
        float dist = std::abs(normalized - NF4_CODEBOOK[i]);
        if (dist < best_dist) { best_dist = dist; best = i; }
    }
    return (uint8_t)best;
}

float nf4_dequantize(uint8_t idx, float scale) {
    if (idx >= 16) idx = 0;
    return NF4_CODEBOOK[idx] * scale;
}

Tensor nf4_dequant_tensor(const uint8_t* data, const float* scales,
                          int64_t n, int64_t block_size) {
    Tensor out({n});
    float* od = out.data<float>();
#ifdef OIL_HAS_AVX2
    if (n >= 8) {
        dequant_tensor_nf4_avx2(data, scales, od, n, block_size, NF4_CODEBOOK);
        return out;
    }
#endif
    for (int64_t i = 0; i < n; ++i) {
        int64_t block_idx = i / block_size;
        od[i] = nf4_dequantize(data[i], scales[block_idx]);
    }
    return out;
}

// ===========================================================================
// Roundtrip test helpers
// ===========================================================================

float compute_quant_error(const Tensor& original, const Tensor& dequantized) {
    int64_t n = original.numel();
    float max_err = 0;
    const float* od = original.data<float>();
    const float* dd = dequantized.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        float err = std::abs(od[i] - dd[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ===========================================================================
// FP8 E4M3: Batch operations and extensions
// ===========================================================================

Tensor fp8_e4m3_quantize_tensor(const float* data, int64_t n) {
    Tensor out({n}, oil::DType::U8);
    uint8_t* od = out.data<uint8_t>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e4m3_quantize(data[i]);
    return out;
}

Tensor fp8_e4m3_quant_gemm(const Tensor& a, const uint8_t* b_q,
                           int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
#ifdef OIL_HAS_AVX2
    quant_gemm_fp8_avx2(ad, cd, b_q, M, N, K, false);
#else
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * fp8_e4m3_dequantize(b_q[k * N + n]);
        }
    }
#endif
    return C;
}

void fp8_e4m3_quantize_per_channel(const Tensor& t, int channel_dim,
                                    Tensor& q, Tensor& scales) {
    OIL_CHECK(t.rank() == 2, "fp8_e4m3_quantize_per_channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    q = Tensor(t.shape(), oil::DType::U8);
    scales = Tensor({channels});
    const float* td = t.data<float>();
    uint8_t* qd = q.data<uint8_t>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        if (max_abs < 1e-10f) max_abs = 1.0f;
        sd[c] = max_abs;
        float inv_scale = 448.0f / max_abs;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            qd[idx] = fp8_e4m3_quantize(td[idx] * inv_scale);
        }
    }
}

void fp8_e4m3_dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                      int channel_dim, Tensor& out) {
    OIL_CHECK(q.rank() == 2, "fp8_e4m3_dequantize_per_channel expects 2D q tensor");
    out = Tensor(q.shape());
    int64_t d0 = q.dim(0), d1 = q.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    const uint8_t* qd = q.data<uint8_t>();
    const float* sd = scales.data<float>();
    float* od = out.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float mul = sd[c] / 448.0f;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            od[idx] = fp8_e4m3_dequantize(qd[idx]) * mul;
        }
    }
}

float fp8_e4m3_quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float fp8_e4m3_quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

// ===========================================================================
// FP8 E5M2: Batch operations and extensions
// ===========================================================================

Tensor fp8_e5m2_quantize_tensor(const float* data, int64_t n) {
    Tensor out({n}, oil::DType::U8);
    uint8_t* od = out.data<uint8_t>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e5m2_quantize(data[i]);
    return out;
}

Tensor fp8_e5m2_quant_gemm(const Tensor& a, const uint8_t* b_q,
                           int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
#ifdef OIL_HAS_AVX2
    quant_gemm_fp8_avx2(ad, cd, b_q, M, N, K, true);
#else
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * fp8_e5m2_dequantize(b_q[k * N + n]);
        }
    }
#endif
    return C;
}

void fp8_e5m2_quantize_per_channel(const Tensor& t, int channel_dim,
                                    Tensor& q, Tensor& scales) {
    OIL_CHECK(t.rank() == 2, "fp8_e5m2_quantize_per_channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    q = Tensor(t.shape(), oil::DType::U8);
    scales = Tensor({channels});
    const float* td = t.data<float>();
    uint8_t* qd = q.data<uint8_t>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        if (max_abs < 1e-10f) max_abs = 1.0f;
        sd[c] = max_abs;
        float inv_scale = 57344.0f / max_abs;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            qd[idx] = fp8_e5m2_quantize(td[idx] * inv_scale);
        }
    }
}

void fp8_e5m2_dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                      int channel_dim, Tensor& out) {
    OIL_CHECK(q.rank() == 2, "fp8_e5m2_dequantize_per_channel expects 2D q tensor");
    out = Tensor(q.shape());
    int64_t d0 = q.dim(0), d1 = q.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    const uint8_t* qd = q.data<uint8_t>();
    const float* sd = scales.data<float>();
    float* od = out.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float mul = sd[c] / 57344.0f;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            od[idx] = fp8_e5m2_dequantize(qd[idx]) * mul;
        }
    }
}

float fp8_e5m2_quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float fp8_e5m2_quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

// ===========================================================================
// NF4: Batch operations and extensions
// ===========================================================================

Tensor nf4_quantize_tensor(const float* data, int64_t n, int64_t block_size) {
    int64_t num_blocks = (n + block_size - 1) / block_size;
    int64_t scale_bytes = num_blocks * (int64_t)sizeof(float);
    Tensor out({scale_bytes + n}, oil::DType::U8);
    uint8_t* base = out.data<uint8_t>();
    float* sd = reinterpret_cast<float*>(base);
    uint8_t* od = base + scale_bytes;
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * block_size;
        int64_t end = std::min(start + block_size, n);
        float max_abs = 0;
        for (int64_t i = start; i < end; ++i)
            max_abs = std::max(max_abs, std::abs(data[i]));
        sd[b] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = start; i < end; ++i)
            od[i] = nf4_quantize(data[i], max_abs);
    }
    return out;
}

Tensor nf4_quant_gemm(const Tensor& a, const uint8_t* b_q, const float* scales,
                      int64_t M, int64_t N, int64_t K, int64_t block_size) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
#ifdef OIL_HAS_AVX2
    quant_gemm_nf4_avx2(ad, cd, b_q, scales, M, N, K, block_size, NF4_CODEBOOK);
#else
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n_val = 0; n_val < N; ++n_val) {
                int64_t flat = k * N + n_val;
                int64_t block = flat / block_size;
                cd[m * N + n_val] += a_val * nf4_dequantize(b_q[flat], scales[block]);
            }
        }
    }
#endif
    return C;
}

void nf4_quantize_per_channel(const Tensor& t, int channel_dim,
                               int64_t block_size, Tensor& q, Tensor& scales) {
    OIL_CHECK(t.rank() == 2, "nf4_quantize_per_channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    q = Tensor(t.shape(), oil::DType::U8);
    int64_t num_scales = (t.numel() + block_size - 1) / block_size;
    scales = Tensor({num_scales});
    const float* td = t.data<float>();
    uint8_t* qd = q.data<uint8_t>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        int64_t chan_start = (channel_dim == 0) ? c * other : c;
        for (int64_t off = 0; off < other; off += block_size) {
            int64_t block_end = std::min(off + block_size, other);
            float max_abs = 0;
            for (int64_t i = off; i < block_end; ++i) {
                int64_t idx = (channel_dim == 0) ? chan_start + i : i * channels + c;
                max_abs = std::max(max_abs, std::abs(td[idx]));
            }
            if (max_abs < 1e-10f) max_abs = 1.0f;
            int64_t block_idx = (c * other + off) / block_size;
            sd[block_idx] = max_abs;
            for (int64_t i = off; i < block_end; ++i) {
                int64_t idx = (channel_dim == 0) ? chan_start + i : i * channels + c;
                qd[idx] = nf4_quantize(td[idx], max_abs);
            }
        }
    }
}

void nf4_dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                 int64_t block_size, int channel_dim, Tensor& out) {
    out = Tensor(q.shape());
    const uint8_t* qd = q.data<uint8_t>();
    const float* sd = scales.data<float>();
    float* od = out.data<float>();
    int64_t n = q.numel();
    for (int64_t i = 0; i < n; ++i) {
        int64_t block = i / block_size;
        od[i] = nf4_dequantize(qd[i], sd[block]);
    }
}

float nf4_quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float nf4_quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

// ===========================================================================
// Roundtrip test helpers
// ===========================================================================

float compute_quant_mse(const Tensor& original, const Tensor& reconstructed) {
    int64_t n = original.numel();
    float mse = 0;
    const float* od = original.data<float>();
    const float* rd = reconstructed.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        float diff = od[i] - rd[i];
        mse += diff * diff;
    }
    return mse / (float)n;
}

float compute_quant_snr(const Tensor& original, const Tensor& reconstructed) {
    int64_t n = original.numel();
    float signal = 0, noise = 0;
    const float* od = original.data<float>();
    const float* rd = reconstructed.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        signal += od[i] * od[i];
        float diff = od[i] - rd[i];
        noise += diff * diff;
    }
    if (noise < 1e-30f) return 100.0f;
    return 10.0f * std::log10(signal / noise);
}

// ===========================================================================
// AVX2 SIMD-accelerated quantize/dequantize batch operations
// ===========================================================================
#ifdef OIL_HAS_AVX2

static void dequant_tensor_nf4_avx2(const uint8_t* data, const float* scales,
                                     float* out, int64_t n, int64_t block_size,
                                     const float* nf4_lut) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i idx8 = _mm_loadl_epi64((const __m128i*)(data + i));
        __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
        __m256 vals = _mm256_i32gather_ps(nf4_lut, idx32, 4);
        __m256i blk = _mm256_setr_epi32(
            (int)(i / block_size), (int)((i + 1) / block_size),
            (int)((i + 2) / block_size), (int)((i + 3) / block_size),
            (int)((i + 4) / block_size), (int)((i + 5) / block_size),
            (int)((i + 6) / block_size), (int)((i + 7) / block_size));
        __m256 scl = _mm256_i32gather_ps(scales, blk, 4);
        vals = _mm256_mul_ps(vals, scl);
        _mm256_storeu_ps(out + i, vals);
    }
    for (; i < n; ++i) {
        int64_t blk = i / block_size;
        out[i] = nf4_lut[data[i]] * scales[blk];
    }
}

static void dequant_tensor_fp8_avx2(const uint8_t* data, float* out,
                                     int64_t n, bool e5m2) {
    static float lut_e4m3[256];
    static float lut_e5m2[256];
    static std::once_flag lut_flag;
    std::call_once(lut_flag, []() {
        for (int i = 0; i < 256; ++i) {
            lut_e4m3[i] = fp8_e4m3_dequantize((uint8_t)i);
            lut_e5m2[i] = fp8_e5m2_dequantize((uint8_t)i);
        }
    });
    const float* lut = e5m2 ? lut_e5m2 : lut_e4m3;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i idx8 = _mm_loadl_epi64((const __m128i*)(data + i));
        __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
        __m256 val = _mm256_i32gather_ps(lut, idx32, 4);
        _mm256_storeu_ps(out + i, val);
    }
    for (; i < n; ++i)
        out[i] = lut[data[i]];
}

// AVX2 quant_gemm: FP8 E4M3 / E5M2 (256-entry LUT)
static void quant_gemm_fp8_avx2(const float* ad, float* cd,
                                 const uint8_t* b_q,
                                 int64_t M, int64_t N, int64_t K,
                                 bool e5m2) {
    static float lut_e4m3[256];
    static float lut_e5m2[256];
    static std::once_flag lut_flag;
    std::call_once(lut_flag, []() {
        for (int i = 0; i < 256; ++i) {
            lut_e4m3[i] = fp8_e4m3_dequantize((uint8_t)i);
            lut_e5m2[i] = fp8_e5m2_dequantize((uint8_t)i);
        }
    });
    const float* lut = e5m2 ? lut_e5m2 : lut_e4m3;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            __m256 a_v = _mm256_set1_ps(a_val);
            int64_t n = 0;
            for (; n + 8 <= N; n += 8) {
                __m128i idx8 = _mm_loadl_epi64(
                    (const __m128i*)(b_q + k * N + n));
                __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
                __m256 b_v = _mm256_i32gather_ps(lut, idx32, 4);
                __m256 c_v = _mm256_loadu_ps(cd + m * N + n);
                _mm256_storeu_ps(cd + m * N + n,
                                 _mm256_fmadd_ps(a_v, b_v, c_v));
            }
            for (; n < N; ++n)
                cd[m * N + n] += a_val * lut[b_q[k * N + n]];
        }
    }
}

// AVX2 quant_gemm: NF4 (per-block scale, 16-entry NF4 LUT)
static void quant_gemm_nf4_avx2(const float* ad, float* cd,
                                 const uint8_t* b_q, const float* scales,
                                 int64_t M, int64_t N, int64_t K,
                                 int64_t block_size, const float* nf4_lut) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            __m256 a_v = _mm256_set1_ps(a_val);
            int64_t n = 0;
            for (; n + 8 <= N; n += 8) {
                int64_t flat_base = k * N + n;
                __m128i idx8 = _mm_loadl_epi64(
                    (const __m128i*)(b_q + flat_base));
                __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
                __m256 vals = _mm256_i32gather_ps(nf4_lut, idx32, 4);
                __m256i blk = _mm256_setr_epi32(
                    (int)(flat_base / block_size),
                    (int)((flat_base + 1) / block_size),
                    (int)((flat_base + 2) / block_size),
                    (int)((flat_base + 3) / block_size),
                    (int)((flat_base + 4) / block_size),
                    (int)((flat_base + 5) / block_size),
                    (int)((flat_base + 6) / block_size),
                    (int)((flat_base + 7) / block_size));
                __m256 scl = _mm256_i32gather_ps(scales, blk, 4);
                __m256 b_v = _mm256_mul_ps(vals, scl);
                __m256 c_v = _mm256_loadu_ps(cd + m * N + n);
                _mm256_storeu_ps(cd + m * N + n,
                                 _mm256_fmadd_ps(a_v, b_v, c_v));
            }
            for (; n < N; ++n) {
                int64_t flat = k * N + n;
                int64_t blk = flat / block_size;
                cd[m * N + n] += a_val * nf4_lut[b_q[flat]] * scales[blk];
            }
        }
    }
}

#endif // OIL_HAS_AVX2

} // namespace engines
} // namespace oil
