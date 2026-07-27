#include "oil/oil_engines.h"
#include "oil/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace oil {
namespace engines {

// ===========================================================================
// Oil1 Engine: {-1, +1} with per-tensor scale
// ===========================================================================

Oil1Engine::Oil1Engine() {}

Tensor Oil1Engine::quantize(const Tensor& weight) {
    int64_t n = weight.numel();
    int64_t packed_size = (n + 7) / 8;
    int64_t packed_floats = (packed_size + 3) / 4;
    int64_t hdr = 2;
    Tensor out({hdr + packed_floats});
    out.zero_();
    float* rd = out.data<float>();
    rd[0] = (float)n;
    uint8_t* pd = reinterpret_cast<uint8_t*>(rd + hdr);
    const float* wd = weight.data<float>();
    float max_abs = 0;
    for (int64_t i = 0; i < n; ++i)
        max_abs = std::max(max_abs, std::abs(wd[i]));
    rd[1] = max_abs;
    if (max_abs < 1e-10f) max_abs = 1.0f;
    for (int64_t i = 0; i < n; ++i) {
        uint8_t bit = (wd[i] >= 0.0f) ? 1 : 0;
        int64_t byte_idx = i / 8;
        int bit_off = (int)(i % 8);
        pd[byte_idx] |= (bit << bit_off);
    }
    return out;
}

Tensor Oil1Engine::dequantize(const Tensor& packed, float scale, int64_t n) {
    const float* rd = packed.data<float>();
    int64_t hdr = 2;
    float use_scale = scale;
    int64_t use_n = n;
    int64_t packed_size_expected = (n + 7) / 8;
    int64_t packed_floats_expected = (packed_size_expected + 3) / 4;
    if (packed.numel() >= hdr + packed_floats_expected && rd[0] > 0) {
        use_n = (int64_t)rd[0];
        use_scale = rd[1];
    }
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(rd + hdr);
    Tensor out({use_n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < use_n; ++i) {
        int64_t byte_idx = i / 8;
        int bit_off = (int)(i % 8);
        uint8_t bit = (pd[byte_idx] >> bit_off) & 1;
        od[i] = (bit == 1) ? use_scale : -use_scale;
    }
    return out;
}

// ===========================================================================
// Oil1 Engine: Extensions
// ===========================================================================

Tensor Oil1Engine::quantize_batch(const Tensor& t) {
    int64_t n = t.numel();
    int64_t packed_size = (n + 7) / 8;
    int64_t hdr = 2;
    int64_t packed_floats = (packed_size + 3) / 4;
    Tensor out({hdr + packed_floats + 1});
    out.zero_();
    float* rd = out.data<float>();
    rd[0] = (float)n;
    rd[1] = (float)packed_size;
    uint8_t* pd = reinterpret_cast<uint8_t*>(rd + hdr);
    const float* wd = t.data<float>();
    float max_abs = 0;
    for (int64_t i = 0; i < n; ++i)
        max_abs = std::max(max_abs, std::abs(wd[i]));
    rd[hdr + packed_floats] = max_abs;
    if (max_abs < 1e-10f) max_abs = 1.0f;
    for (int64_t i = 0; i < n; ++i) {
        uint8_t bit = (wd[i] >= 0.0f) ? 1 : 0;
        int64_t byte_idx = i / 8;
        int bit_off = (int)(i % 8);
        pd[byte_idx] |= (bit << bit_off);
    }
    return out;
}

Tensor Oil1Engine::dequantize_batch(const Tensor& q) {
    const float* rd = q.data<float>();
    int64_t n = (int64_t)rd[0];
    int64_t packed_size = (int64_t)rd[1];
    int64_t hdr = 2;
    int64_t packed_floats = (packed_size + 3) / 4;
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(rd + hdr);
    float scale = rd[hdr + packed_floats];
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t byte_idx = i / 8;
        int bit_off = (int)(i % 8);
        uint8_t bit = (pd[byte_idx] >> bit_off) & 1;
        od[i] = (bit == 1) ? scale : -scale;
    }
    return out;
}

Tensor Oil1Engine::quant_gemm(const Tensor& a, const Tensor& b_packed,
                                 float b_scale, int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(b_packed.data<float>());
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n) {
                int64_t flat = k * N + n;
                int64_t byte_idx = flat / 8;
                int bit_off = (int)(flat % 8);
                uint8_t bit = (pd[byte_idx] >> bit_off) & 1;
                float b_val = (bit == 1) ? b_scale : -b_scale;
                cd[m * N + n] += a_val * b_val;
            }
        }
    }
    return C;
}

void Oil1Engine::quantize_per_channel(const Tensor& t, int channel_dim,
                                          Tensor& q, Tensor& scales) {
    OIL_CHECK(t.rank() == 2, "Oil1 per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    int64_t n = t.numel();
    int64_t packed_size = (n + 7) / 8;
    q = Tensor({packed_size});
    q.zero_();
    scales = Tensor({channels});
    const float* td = t.data<float>();
    uint8_t* pd = reinterpret_cast<uint8_t*>(q.data<float>());
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        sd[c] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            uint8_t bit = (td[idx] >= 0.0f) ? 1 : 0;
            int64_t byte_idx = idx / 8;
            int bit_off = (int)(idx % 8);
            pd[byte_idx] |= (bit << bit_off);
        }
    }
}

void Oil1Engine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                            int channel_dim, Tensor& out) {
    int64_t n = scales.numel();
    int64_t d0 = n, d1 = 1;
    if (channel_dim == 0) { d0 = n; d1 = q.numel() * 8 / n; }
    else { d1 = n; d0 = q.numel() * 8 / n; }
    out = Tensor({d0, d1});
    float* od = out.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(q.data<float>());
    const float* sd = scales.data<float>();
    for (int64_t c = 0; c < n; ++c) {
        float scale = sd[c];
        int64_t other = out.numel() / n;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * n + c;
            int64_t byte_idx = idx / 8;
            int bit_off = (int)(idx % 8);
            uint8_t bit = (pd[byte_idx] >> bit_off) & 1;
            od[idx] = (bit == 1) ? scale : -scale;
        }
    }
}

float Oil1Engine::quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float Oil1Engine::quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

} // namespace engines
} // namespace oil
