#include "oil/oil_engines.h"
#include "oil/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace oil {
namespace engines {

// ===========================================================================
// AWQ: Activation-aware Weight Quantization
// Per-channel scaling + per-group INT4 quantization
// ===========================================================================

AWQQuantizer::AWQQuantizer(int64_t group_size, float alpha)
    : group_size_(group_size), alpha_(alpha) {}

void AWQQuantizer::compute_scales(const Tensor& weight, const Tensor& activation) {
    int64_t N = weight.dim(0);
    int64_t K = weight.dim(1);
    scales_.resize(N, 1.0f);
    const float* wd = weight.data<float>();
    const float* ad = activation.data<float>();
    for (int64_t n = 0; n < N; ++n) {
        float max_w = 0, max_a = 0;
        for (int64_t k = 0; k < K; ++k) {
            max_w = std::max(max_w, std::abs(wd[n * K + k]));
        }
        for (int64_t k = 0; k < K; ++k) {
            max_a = std::max(max_a, std::abs(ad[k]));
        }
        float s = std::pow(max_a, alpha_) / (max_w + 1e-10f);
        s = std::max(1e-8f, std::min(s, 1e4f));
        scales_[(size_t)n] = s;
    }
}

Tensor AWQQuantizer::quantize(const Tensor& weight) {
    int64_t N = weight.dim(0);
    int64_t K = weight.dim(1);
    int64_t num_groups = (K + group_size_ - 1) / group_size_;
    Tensor q_weight({N, K});
    float* qd = q_weight.data<float>();
    const float* wd = weight.data<float>();
    for (int64_t n = 0; n < N; ++n) {
        float s = scales_[(size_t)n];
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t start = g * group_size_;
            int64_t end = std::min(start + group_size_, K);
            float max_abs = 0;
            for (int64_t k = start; k < end; ++k)
                max_abs = std::max(max_abs, std::abs(wd[n * K + k] * s));
            float scale = max_abs / 7.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            for (int64_t k = start; k < end; ++k) {
                int q = (int)std::round(wd[n * K + k] * s / scale);
                q = std::max(-8, std::min(7, q));
                qd[n * K + k] = (float)q * scale / s;
            }
        }
    }
    return q_weight;
}

Tensor AWQQuantizer::dequantize(const Tensor& q_weight) {
    return q_weight;
}

// ===========================================================================
// GPTQ: Quantization via approximate second-order Hessian
// Per-group INT4 quantization with Hessian-based error compensation
// ===========================================================================

GPTQQuantizer::GPTQQuantizer(int64_t group_size, int bits)
    : group_size_(group_size), bits_(bits) {
    max_q_ = (float)((1 << (bits - 1)) - 1);
    min_q_ = -(float)(1 << (bits - 1));
}

Tensor GPTQQuantizer::quantize(const Tensor& weight, const Tensor& hessian) {
    int64_t N = weight.dim(0);
    int64_t K = weight.dim(1);
    Tensor q_weight({N, K});
    float* qd = q_weight.data<float>();
    const float* wd = weight.data<float>();
    const float* hd = hessian.data<float>();
    for (int64_t n = 0; n < N; ++n) {
        int64_t num_groups = (K + group_size_ - 1) / group_size_;
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t start = g * group_size_;
            int64_t end = std::min(start + group_size_, K);
            float scale = 0;
            for (int64_t k = start; k < end; ++k) {
                float h = hd[n * K + k];
                scale = std::max(scale, std::abs(wd[n * K + k]) / (h + 1e-8f));
            }
            scale = std::max(1e-10f, scale / max_q_);
            for (int64_t k = start; k < end; ++k) {
                int q = (int)std::round(wd[n * K + k] / scale);
                q = (int)std::max((float)min_q_, std::min((float)max_q_, (float)q));
                qd[n * K + k] = (float)q * scale;
            }
        }
    }
    return q_weight;
}

Tensor GPTQQuantizer::dequantize(const Tensor& q_weight) {
    return q_weight;
}

// ===========================================================================
// I2S Engine: Int2 + Scale (BitNet compatible)
// Pack 4 ternary values (2-bit each) into 1 byte with shared scale
// ===========================================================================

I2SEngine::I2SEngine(int64_t block_size) : block_size_(block_size) {}

Tensor I2SEngine::quantize(const Tensor& weight) {
    int64_t n = weight.numel();
    int64_t num_blocks = (n + block_size_ - 1) / block_size_;
    int64_t packed_size = (n + 3) / 4;
    Tensor packed({packed_size});
    Tensor scales({num_blocks});
    packed.zero_();
    const float* wd = weight.data<float>();
    uint8_t* pd = reinterpret_cast<uint8_t*>(packed.data<float>());
    float* sd = scales.data<float>();
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * block_size_;
        int64_t end = std::min(start + block_size_, n);
        float max_abs = 0;
        for (int64_t i = start; i < end; ++i)
            max_abs = std::max(max_abs, std::abs(wd[i]));
        float scale = max_abs;
        sd[b] = scale;
        if (scale < 1e-10f) scale = 1.0f;
        for (int64_t i = start; i < end; ++i) {
            float normalized = wd[i] / scale;
            int8_t ternary;
            if (normalized > 0.33f) ternary = 1;
            else if (normalized < -0.33f) ternary = -1;
            else ternary = 0;
            uint8_t code = (uint8_t)((ternary + 1) & 0x3);
            int64_t byte_idx = i / 4;
            int bit_off = (int)(i % 4) * 2;
            if (bit_off == 0)
                pd[byte_idx] = code;
            else
                pd[byte_idx] |= (code << bit_off);
        }
    }
    return packed;
}

Tensor I2SEngine::dequantize(const Tensor& packed, const Tensor& scales, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(packed.data<float>());
    const float* sd = scales.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t block_idx = i / block_size_;
        float scale = sd[block_idx];
        int64_t byte_idx = i / 4;
        int bit_off = (int)(i % 4) * 2;
        uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
        float ternary = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
        od[i] = ternary * scale;
    }
    return out;
}

// ===========================================================================
// AWQ Quantizer: Extensions
// ===========================================================================

Tensor AWQQuantizer::quantize_batch(const Tensor& t) {
    return quantize(t);
}

Tensor AWQQuantizer::dequantize_batch(const Tensor& q) {
    return dequantize(q);
}

Tensor AWQQuantizer::quant_gemm(const Tensor& a, const Tensor& b_q,
                                 int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    const float* bd = b_q.data<float>();
    float* cd = C.data<float>();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * bd[k * N + n];
        }
    }
    return C;
}

void AWQQuantizer::quantize_per_channel(const Tensor& t, int channel_dim,
                                         Tensor& q, Tensor& scales) {
    OIL_CHECK(t.rank() == 2, "AWQ per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    q = Tensor(t.shape());
    scales = Tensor({channels});
    const float* td = t.data<float>();
    float* qd = q.data<float>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        if (max_abs < 1e-10f) max_abs = 1.0f;
        sd[c] = max_abs;
        int64_t num_groups = (other + group_size_ - 1) / group_size_;
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t start = g * group_size_;
            int64_t end = std::min(start + group_size_, other);
            float gmax = 0;
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
                gmax = std::max(gmax, std::abs(td[idx]));
            }
            float group_scale = gmax / 7.0f;
            if (group_scale < 1e-10f) group_scale = 1e-10f;
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
                int qv = (int)std::round(td[idx] / group_scale);
                qv = std::max(-8, std::min(7, qv));
                qd[idx] = (float)qv * group_scale;
            }
        }
    }
}

void AWQQuantizer::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                           int channel_dim, Tensor& out) {
    out = q;
}

float AWQQuantizer::quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float AWQQuantizer::quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

// ===========================================================================
// GPTQ Quantizer: Extensions
// ===========================================================================

Tensor GPTQQuantizer::quantize_batch(const Tensor& t) {
    Tensor dummy_hessian({t.dim(0), t.dim(1)});
    dummy_hessian.fill(1.0f);
    return quantize(t, dummy_hessian);
}

Tensor GPTQQuantizer::dequantize_batch(const Tensor& q) {
    return dequantize(q);
}

Tensor GPTQQuantizer::quant_gemm(const Tensor& a, const Tensor& b_q,
                                  int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    const float* bd = b_q.data<float>();
    float* cd = C.data<float>();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * bd[k * N + n];
        }
    }
    return C;
}

void GPTQQuantizer::quantize_per_channel(const Tensor& t, int channel_dim,
                                          Tensor& q, Tensor& scales) {
    OIL_CHECK(t.rank() == 2, "GPTQ per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    q = Tensor(t.shape());
    scales = Tensor({channels});
    const float* td = t.data<float>();
    float* qd = q.data<float>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        if (max_abs < 1e-10f) max_abs = 1.0f;
        sd[c] = max_abs;
        int64_t num_groups = (other + group_size_ - 1) / group_size_;
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t start = g * group_size_;
            int64_t end = std::min(start + group_size_, other);
            float gmax = 0;
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
                gmax = std::max(gmax, std::abs(td[idx]) / (1.0f + 1e-8f));
            }
            float group_scale = gmax / max_q_;
            if (group_scale < 1e-10f) group_scale = 1e-10f;
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
                int qv = (int)std::round(td[idx] / group_scale);
                qv = (int)std::max(min_q_, std::min(max_q_, (float)qv));
                qd[idx] = (float)qv * group_scale;
            }
        }
    }
}

void GPTQQuantizer::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                            int channel_dim, Tensor& out) {
    out = q;
}

float GPTQQuantizer::quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float GPTQQuantizer::quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

// ===========================================================================
// I2S Engine: Extensions
// ===========================================================================

Tensor I2SEngine::quantize_batch(const Tensor& t) {
    int64_t n = t.numel();
    int64_t num_blocks = (n + block_size_ - 1) / block_size_;
    int64_t packed_size = (n + 3) / 4;
    int64_t hdr = 3;
    int64_t packed_floats = (packed_size + 3) / 4;
    Tensor out({hdr + packed_floats + num_blocks});
    out.zero_();
    float* rd = out.data<float>();
    rd[0] = (float)n;
    rd[1] = (float)block_size_;
    rd[2] = (float)num_blocks;
    uint8_t* pd = reinterpret_cast<uint8_t*>(rd + hdr);
    float* sd = rd + hdr + packed_floats;
    const float* wd = t.data<float>();
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * block_size_;
        int64_t end = std::min(start + block_size_, n);
        float max_abs = 0;
        for (int64_t i = start; i < end; ++i)
            max_abs = std::max(max_abs, std::abs(wd[i]));
        sd[b] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = start; i < end; ++i) {
            float normalized = wd[i] / max_abs;
            int8_t ternary;
            if (normalized > 0.33f) ternary = 1;
            else if (normalized < -0.33f) ternary = -1;
            else ternary = 0;
            uint8_t code = (uint8_t)((ternary + 1) & 0x3);
            int64_t byte_idx = i / 4;
            int bit_off = (int)(i % 4) * 2;
            if (bit_off == 0)
                pd[byte_idx] = code;
            else
                pd[byte_idx] |= (code << bit_off);
        }
    }
    return out;
}

Tensor I2SEngine::dequantize_batch(const Tensor& q) {
    const float* rd = q.data<float>();
    int64_t n = (int64_t)rd[0];
    int64_t block_size = (int64_t)rd[1];
    int64_t packed_size = (n + 3) / 4;
    int64_t hdr = 3;
    int64_t packed_floats = (packed_size + 3) / 4;
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(rd + hdr);
    const float* sd = rd + hdr + packed_floats;
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t block_idx = i / block_size;
        float scale = sd[block_idx];
        int64_t byte_idx = i / 4;
        int bit_off = (int)(i % 4) * 2;
        uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
        float ternary = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
        od[i] = ternary * scale;
    }
    return out;
}

Tensor I2SEngine::quant_gemm(const Tensor& a, const Tensor& b_packed,
                              const Tensor& b_scales, int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(b_packed.data<float>());
    const float* sd = b_scales.data<float>();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n) {
                int64_t flat = k * N + n;
                int64_t block_idx = flat / block_size_;
                float scale = sd[block_idx];
                int64_t byte_idx = flat / 4;
                int bit_off = (int)(flat % 4) * 2;
                uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
                float ternary = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
                cd[m * N + n] += a_val * ternary * scale;
            }
        }
    }
    return C;
}

void I2SEngine::quantize_per_channel(const Tensor& t, int channel_dim,
                                      Tensor& q, Tensor& scales) {
    OIL_CHECK(t.rank() == 2, "I2S per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    int64_t n = t.numel();
    int64_t packed_size = (n + 3) / 4;
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
            float normalized = td[idx] / max_abs;
            int8_t ternary;
            if (normalized > 0.33f) ternary = 1;
            else if (normalized < -0.33f) ternary = -1;
            else ternary = 0;
            uint8_t code = (uint8_t)((ternary + 1) & 0x3);
            int64_t byte_idx = idx / 4;
            int bit_off = (int)(idx % 4) * 2;
            if (bit_off == 0)
                pd[byte_idx] = code;
            else
                pd[byte_idx] |= (code << bit_off);
        }
    }
}

void I2SEngine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                        int channel_dim, Tensor& out) {
    int64_t n = scales.numel();
    int64_t d0 = n, d1 = 1;
    if (channel_dim == 0) { d0 = n; d1 = q.numel() / n; }
    else { d1 = n; d0 = q.numel() / n; }
    out = Tensor({d0, d1});
    float* od = out.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(q.data<float>());
    const float* sd = scales.data<float>();
    for (int64_t c = 0; c < n; ++c) {
        float scale = sd[c];
        int64_t other = q.numel() / n;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * n + c;
            int64_t byte_idx = idx / 4;
            int bit_off = (int)(idx % 4) * 2;
            uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
            float ternary = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
            od[idx] = ternary * scale;
        }
    }
}

float I2SEngine::quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float I2SEngine::quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

} // namespace engines
} // namespace oil
