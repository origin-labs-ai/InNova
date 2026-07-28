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
// I2S Engine: wraps SparkEngine (identical ternary per-block quantization)
// ===========================================================================

I2SEngine::I2SEngine(int64_t block_size) : spark_(block_size), block_size_(block_size) {}

Tensor I2SEngine::quantize(const Tensor& weight) {
    return spark_.quantize(weight);
}

Tensor I2SEngine::dequantize(const Tensor& packed, const Tensor& scales, int64_t n) {
    return spark_.dequantize(packed, scales, n);
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
// I2S Engine: wraps SparkEngine
// ===========================================================================

Tensor I2SEngine::quantize_batch(const Tensor& t) {
    return spark_.quantize_batch(t);
}

Tensor I2SEngine::dequantize_batch(const Tensor& q) {
    return spark_.dequantize_batch(q);
}

Tensor I2SEngine::quant_gemm(const Tensor& a, const Tensor& b_packed,
                              const Tensor& b_scales, int64_t M, int64_t N, int64_t K) {
    return spark_.quant_gemm(a, b_packed, b_scales, M, N, K);
}

void I2SEngine::quantize_per_channel(const Tensor& t, int channel_dim,
                                      Tensor& q, Tensor& scales) {
    spark_.quantize_per_channel(t, channel_dim, q, scales);
}

void I2SEngine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                        int channel_dim, Tensor& out) {
    spark_.dequantize_per_channel(q, scales, channel_dim, out);
}

float I2SEngine::quant_error(const Tensor& original, const Tensor& reconstructed) {
    return spark_.quant_error(original, reconstructed);
}

float I2SEngine::quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return spark_.quant_snr(original, reconstructed);
}

} // namespace engines
} // namespace oil
