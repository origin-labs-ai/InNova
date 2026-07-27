#include "oil/oil_engines.h"
#include "oil/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

namespace oil {
namespace engines {

// ===========================================================================
// OIL8 Engine: 256-entry FP32 codebook, 8-bit indices
// ===========================================================================

OIL8Engine::OIL8Engine() {
    codebook_.resize(256);
    for (int i = 0; i < 256; ++i)
        codebook_[(size_t)i] = (float)(i - 128) * 0.01f;
}

void OIL8Engine::train_codebook(const float* data, int64_t n) {
    for (int iter = 0; iter < 10; ++iter) {
        std::vector<int> counts(256, 0);
        std::vector<double> sums(256, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            int best = 0;
            float best_dist = 1e10f;
            for (int j = 0; j < 256; ++j) {
                float dist = std::abs(data[i] - codebook_[(size_t)j]);
                if (dist < best_dist) { best_dist = dist; best = j; }
            }
            counts[best]++;
            sums[best] += data[i];
        }
        for (int j = 0; j < 256; ++j) {
            if (counts[j] > 0)
                codebook_[(size_t)j] = (float)(sums[j] / counts[j]);
        }
    }
}

void OIL8Engine::train_codebook_per_block(const float* data, int64_t n,
                                           int64_t block_size, int lloyd_iters) {
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;

    // Step 1: collect all normalized weights (divide each block by its abs-max scale)
    std::vector<float> normalized(n);
    std::vector<float> scales(num_blocks);
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        float amax = 0.0f;
        for (int64_t i = start; i < end; i++)
            amax = (std::max)(amax, std::abs(data[i]));
        if (amax < 1e-10f) amax = 1.0f;
        scales[(size_t)b] = amax;
        for (int64_t i = start; i < end; i++)
            normalized[(size_t)i] = data[i] / amax;
    }

    // Step 2: train codebook on normalized data using Lloyd-Max
    // Initialize uniformly across [-1, +1]
    for (int i = 0; i < 256; ++i)
        codebook_[(size_t)i] = -1.0f + 2.0f * (float)i / 255.0f;

    for (int iter = 0; iter < lloyd_iters; ++iter) {
        std::vector<int> counts(256, 0);
        std::vector<double> sums(256, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            int best = 0;
            float best_dist = 1e10f;
            for (int j = 0; j < 256; ++j) {
                float dist = std::abs(normalized[(size_t)i] - codebook_[(size_t)j]);
                if (dist < best_dist) { best_dist = dist; best = j; }
            }
            counts[best]++;
            sums[best] += normalized[(size_t)i];
        }
        bool changed = false;
        for (int j = 0; j < 256; ++j) {
            if (counts[j] > 0) {
                float new_val = (float)(sums[j] / (double)counts[j]);
                if (std::abs(new_val - codebook_[(size_t)j]) > 1e-8f) changed = true;
                codebook_[(size_t)j] = new_val;
            }
        }
        if (!changed) break;
    }
}

void OIL8Engine::quantize_per_block(const float* data, int64_t n, int64_t block_size,
                                     uint8_t* indices_out, float* scales_out) const {
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        int64_t len = end - start;

        float amax = 0.0f;
        for (int64_t i = start; i < end; i++)
            amax = (std::max)(amax, std::abs(data[i]));
        if (amax < 1e-10f) amax = 1.0f;
        scales_out[(size_t)b] = amax;

        for (int64_t i = 0; i < len; i++) {
            float normalized = data[start + i] / amax;
            int best = 0;
            float best_dist = 1e10f;
            for (int j = 0; j < 256; ++j) {
                float dist = std::abs(normalized - codebook_[(size_t)j]);
                if (dist < best_dist) { best_dist = dist; best = j; }
            }
            indices_out[start + i] = (uint8_t)best;
        }
    }
}

void OIL8Engine::dequantize_per_block(const uint8_t* indices, const float* scales,
                                       int64_t n, int64_t block_size, float* out) const {
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        float scale = scales[(size_t)b];
        for (int64_t i = start; i < end; i++) {
            out[i] = codebook_[(size_t)indices[i]] * scale;
        }
    }
}

uint8_t OIL8Engine::quantize(float val) const {
    if (!use_stochastic_) {
        int best = 0;
        float best_dist = 1e10f;
        for (int i = 0; i < 256; ++i) {
            float dist = std::abs(val - codebook_[(size_t)i]);
            if (dist < best_dist) { best_dist = dist; best = i; }
        }
        return (uint8_t)best;
    }
    // Stochastic rounding: weight entries by proximity, sample randomly
    // This produces zero-mean quantization noise (unlike argmin which is biased)
    float total = 0.0f;
    float weights[256];
    float inv_temp = 1.0f / stoch_temperature_;
    for (int i = 0; i < 256; ++i) {
        float d = std::abs(val - codebook_[(size_t)i]);
        weights[i] = std::exp(-d * inv_temp);
        total += weights[i];
    }
    if (total < 1e-30f) total = 1.0f;
    static thread_local std::mt19937 rng(42);
    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) * total;
    float cum = 0.0f;
    for (int i = 0; i < 256; ++i) {
        cum += weights[i];
        if (r <= cum) return (uint8_t)i;
    }
    return 255;
}

float OIL8Engine::dequantize(uint8_t idx) const {
    return codebook_[(size_t)idx];
}

Tensor OIL8Engine::dequant_tensor(const uint8_t* indices, int64_t n) const {
    Tensor out({n});
    float* od = out.data<float>();
#ifdef OIL_HAS_AVX2
    if (n >= 8) {
        dequant_tensor_oil8_avx2(indices, od, n, codebook_.data());
        return out;
    }
#endif
    for (int64_t i = 0; i < n; ++i)
        od[i] = dequantize(indices[i]);
    return out;
}

// ===========================================================================
// OIL8 Engine: Extensions
// ===========================================================================

Tensor OIL8Engine::quantize_tensor(const float* data, int64_t n) const {
    Tensor out({n}, oil::DType::U8);
    uint8_t* od = out.data<uint8_t>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = quantize(data[i]);
    return out;
}

Tensor OIL8Engine::quant_gemm(const Tensor& a, const uint8_t* b_idx,
                               int64_t M, int64_t N, int64_t K) const {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
#ifdef OIL_HAS_AVX2
    quant_gemm_oil8_avx2(ad, cd, b_idx, M, N, K, codebook_.data());
#else
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * dequantize(b_idx[k * N + n]);
        }
    }
#endif
    return C;
}

void OIL8Engine::quantize_per_channel(const Tensor& t, int channel_dim,
                                       Tensor& q, Tensor& scales) const {
    OIL_CHECK(t.rank() == 2, "OIL8 per-channel expects 2D tensor");
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
        sd[c] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            float normalized = td[idx] / max_abs;
            qd[idx] = quantize(normalized);
        }
    }
}

void OIL8Engine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                         int channel_dim, Tensor& out) const {
    OIL_CHECK(q.rank() == 2, "OIL8 dequantize_per_channel expects 2D q tensor");
    out = Tensor(q.shape());
    int64_t d0 = q.dim(0), d1 = q.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    const uint8_t* qd = q.data<uint8_t>();
    const float* sd = scales.data<float>();
    float* od = out.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float scale = sd[c];
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            od[idx] = dequantize(qd[idx]) * scale;
        }
    }
}

float OIL8Engine::quant_error(const Tensor& original, const Tensor& reconstructed) const {
    return compute_quant_mse(original, reconstructed);
}

float OIL8Engine::quant_snr(const Tensor& original, const Tensor& reconstructed) const {
    return compute_quant_snr(original, reconstructed);
}

} // namespace engines
} // namespace oil
