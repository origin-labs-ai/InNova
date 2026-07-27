#include "oil/native_weight.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <memory>

namespace oil {
namespace native {

// ─── OIL4Codebook ───────────────────────────────────────────────

OIL4Codebook::OIL4Codebook() {
    for (size_t i = 0; i < K; i++) {
        centroids_[i] = -1.0f + (2.0f * i + 1.0f) / K;
    }
}

OIL4Codebook::OIL4Codebook(const float* centroids) {
    std::memcpy(centroids_, centroids, K * sizeof(float));
}

void OIL4Codebook::set_quantile_spacing(const float* data, size_t n) {
    std::vector<float> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < K; i++) {
        size_t idx = (size_t)((double)(i + 1) * n / (K + 1));
        if (idx >= n) idx = n - 1;
        centroids_[i] = sorted[idx];
    }
}

uint8_t OIL4Codebook::nearest(float val) const {
    uint8_t best = 0;
    float best_dist = std::abs(val - centroids_[0]);
    for (size_t i = 1; i < K; i++) {
        float d = std::abs(val - centroids_[i]);
        if (d < best_dist) { best_dist = d; best = (uint8_t)i; }
    }
    return best;
}

void OIL4Codebook::train(const float* data, size_t n, int iterations) {
    set_quantile_spacing(data, n);
    if (n < K || iterations < 2) return;

    std::vector<uint8_t> assign(n);
    std::vector<double> sums(K);
    std::vector<int> counts(K);

    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < n; i++) {
            uint8_t best = 0;
            float best_dist = std::abs(data[i] - centroids_[0]);
            for (size_t c = 1; c < K; c++) {
                float d = std::abs(data[i] - centroids_[c]);
                if (d < best_dist) { best_dist = d; best = (uint8_t)c; }
            }
            assign[i] = best;
        }
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n; i++) { sums[assign[i]] += data[i]; counts[assign[i]]++; }
        bool changed = false;
        for (size_t c = 0; c < K; c++) {
            if (counts[c] > 0) {
                float new_val = (float)(sums[c] / (double)counts[c]);
                if (std::abs(new_val - centroids_[c]) > 1e-7f) changed = true;
                centroids_[c] = new_val;
            }
        }
        if (!changed) break;
    }
}

// ─── OIL8Codebook ───────────────────────────────────────────────

OIL8Codebook::OIL8Codebook() {
    // Default: uniform spacing over [-1, 1]
    for (size_t i = 0; i < K; i++) {
        centroids_[i] = -1.0f + (2.0f * i + 1.0f) / K;
    }
}

OIL8Codebook::OIL8Codebook(const float* centroids) {
    std::memcpy(centroids_, centroids, K * sizeof(float));
}

void OIL8Codebook::set_quantile_spacing(const float* data, size_t n) {
    std::vector<float> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < K; i++) {
        size_t idx = (size_t)((double)(i + 1) * n / (K + 1));
        if (idx >= n) idx = n - 1;
        centroids_[i] = sorted[idx];
    }
}

uint8_t OIL8Codebook::nearest(float val) const {
    uint8_t best = 0;
    float best_dist = std::abs(val - centroids_[0]);
    for (size_t i = 1; i < K; i++) {
        float d = std::abs(val - centroids_[i]);
        if (d < best_dist) { best_dist = d; best = (uint8_t)i; }
    }
    return best;
}

void OIL8Codebook::train(const float* data, size_t n, int iterations) {
    set_quantile_spacing(data, n);
    if (n < K || iterations < 2) return;

    std::vector<uint8_t> assign(n);
    std::vector<double> sums(K);
    std::vector<int> counts(K);

    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < n; i++) {
            uint8_t best = 0;
            float best_dist = std::abs(data[i] - centroids_[0]);
            for (size_t c = 1; c < K; c++) {
                float d = std::abs(data[i] - centroids_[c]);
                if (d < best_dist) { best_dist = d; best = (uint8_t)c; }
            }
            assign[i] = best;
        }
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n; i++) { sums[assign[i]] += data[i]; counts[assign[i]]++; }
        bool changed = false;
        for (size_t c = 0; c < K; c++) {
            if (counts[c] > 0) {
                float new_val = (float)(sums[c] / (double)counts[c]);
                if (std::abs(new_val - centroids_[c]) > 1e-7f) changed = true;
                centroids_[c] = new_val;
            }
        }
        if (!changed) break;
    }
}

OIL8Codebook& NativeOILWeightStore::global_codebook() {
    static OIL8Codebook cb;
    return cb;
}

OIL4Codebook& NativeOILWeightStore::global_oil4_codebook() {
    static OIL4Codebook cb;
    return cb;
}

// ─── OILWeight ──────────────────────────────────────────────────

float OILWeight::dequantize() const {
    switch (fmt) {
        case NativeFormat::OIL1: return scale * oil1_value(idx);
        case NativeFormat::OIL4:    return scale * NativeOILWeightStore::global_oil4_codebook().centroid(idx);
        case NativeFormat::OIL8:    return scale * NativeOILWeightStore::global_codebook().centroid(idx);
        default: return 0.0f;
    }
}

uint8_t OILWeight::quantize(float val) const {
    switch (fmt) {
        case NativeFormat::OIL1: {
            float norm = val / scale;
            if (norm > 0.5f) return 2;
            if (norm < -0.5f) return 0;
            return 1;
        }
        case NativeFormat::OIL4: {
            return NativeOILWeightStore::global_oil4_codebook().nearest(val / scale);
        }
        case NativeFormat::OIL8: {
            return NativeOILWeightStore::global_codebook().nearest(val / scale);
        }
        default: return 0;
    }
}

// ─── OILBlock ───────────────────────────────────────────────────

OILBlock::OILBlock(NativeFormat f, size_t B)
    : indices(std::make_unique<uint8_t[]>(B)), fmt(f), block_size(B) {
    std::memset(indices.get(), 0, B * sizeof(uint8_t));
    scale = 1.0f;
}

float OILBlock::dequantize(size_t i) const {
    switch (fmt) {
        case NativeFormat::OIL1: return scale * oil1_value(indices[i]);
        case NativeFormat::OIL4:    return scale * NativeOILWeightStore::global_oil4_codebook().centroid(indices[i]);
        case NativeFormat::OIL8:    return scale * NativeOILWeightStore::global_codebook().centroid(indices[i]);
        default: return 0.0f;
    }
}

uint8_t OILBlock::quantize(size_t i, float val) const {
    float norm = val / scale;
    switch (fmt) {
        case NativeFormat::OIL1: {
            if (norm > 0.5f) return 2;
            if (norm < -0.5f) return 0;
            return 1;
        }
        case NativeFormat::OIL4: {
            return NativeOILWeightStore::global_oil4_codebook().nearest(norm);
        }
        case NativeFormat::OIL8: {
            return NativeOILWeightStore::global_codebook().nearest(norm);
        }
        default: return 0;
    }
}

// ─── NativeOILWeightStore ───────────────────────────────────────

NativeOILWeightStore::NativeOILWeightStore(size_t num_weights, size_t block_size)
    : num_weights_(num_weights), block_size_(block_size)
    , formats_(std::make_unique<NativeFormat[]>((num_weights + block_size - 1) / block_size))
    , indices_(std::make_unique<uint8_t[]>(num_weights))
    , block_scales_(std::make_unique<float[]>((num_weights + block_size - 1) / block_size))
    , frozen_flag_(std::make_unique<bool[]>(num_weights)) {
    num_blocks_ = (num_weights_ + block_size_ - 1) / block_size_;
    std::memset(indices_.get(), 0, num_weights * sizeof(uint8_t));
    std::memset(frozen_flag_.get(), 0, num_weights * sizeof(bool));
    for (size_t b = 0; b < num_blocks_; b++) { formats_[b] = NativeFormat::OIL1; block_scales_[b] = 1.0f; }
}

void NativeOILWeightStore::initialize(const float* fp32_weights, const float* sensitivity,
                                       float frac_oil8, float frac_spark) {
    reallocate_by_sensitivity(sensitivity, frac_oil8, frac_spark);
    convert_from_fp32(fp32_weights);
}

void NativeOILWeightStore::convert_from_fp32(const float* src) {
    for (size_t i = 0; i < num_weights_; i++) {
        size_t b = i / block_size_;
        size_t off = i % block_size_;
        NativeFormat fmt = formats_[b];
        // Initial scale = max absolute value in block (for SPARK/OIL1)
        // For OIL8: scale = range of weights in block
        if (off == 0) {
            // Compute block scale
            size_t block_end = std::min(num_weights_, (b + 1) * block_size_);
            float max_abs = 0.0f;
            for (size_t j = i; j < block_end; j++) {
                max_abs = std::max(max_abs, std::abs(src[j]));
            }
            block_scales_[b] = (max_abs > 1e-10f) ? max_abs : 1.0f;
        }
        // Quantize weight to index
        float norm = src[i] / block_scales_[b];
        switch (fmt) {
            case NativeFormat::OIL1: {
                if (norm > 0.5f) indices_[i] = 2;
                else if (norm < -0.5f) indices_[i] = 0;
                else indices_[i] = 1;
                break;
            }
            case NativeFormat::OIL4: {
                indices_[i] = global_oil4_codebook().nearest(norm);
                break;
            }
            case NativeFormat::OIL8: {
                indices_[i] = global_codebook().nearest(norm);
                break;
            }
        }
    }
}

void NativeOILWeightStore::dequantize(float* dst) const {
    for (size_t i = 0; i < num_weights_; i++) {
        size_t b = i / block_size_;
        float s = block_scales_[b];
        NativeFormat fmt = formats_[b];
        switch (fmt) {
            case NativeFormat::OIL1: dst[i] = s * oil1_value(indices_[i]); break;
            case NativeFormat::OIL4:    dst[i] = s * global_oil4_codebook().centroid(indices_[i]); break;
            case NativeFormat::OIL8:    dst[i] = s * global_codebook().centroid(indices_[i]); break;
        }
    }
}

void NativeOILWeightStore::apply_oil_update(const float* grad, float lr_scale, float lr_weight) {
    for (size_t i = 0; i < num_weights_; i++) {
        size_t b = i / block_size_;
        float s = block_scales_[b];
        NativeFormat fmt = formats_[b];

        float cv;
        switch (fmt) {
            case NativeFormat::OIL1: cv = oil1_value(indices_[i]); break;
            case NativeFormat::OIL4:    cv = global_oil4_codebook().centroid(indices_[i]); break;
            case NativeFormat::OIL8:    cv = global_codebook().centroid(indices_[i]); break;
        }
        float w = s * cv;
        float g = grad[i];

        // Scale update — normalized by block_size (shared scale)
        float grad_scale = g * cv;
        block_scales_[b] -= lr_scale * grad_scale / (float)block_size_;
        if (block_scales_[b] < 1e-10f) block_scales_[b] = 1e-10f;

        // Index update (Theorem 5d.3 — virtual continuous step + dead zone)
        if (!frozen_flag_[i]) {
            float w_virtual = w - lr_weight * g;
            float dz = dead_zone_radius(fmt, s);
            float delta = std::abs(w_virtual - w);
            if (delta > dz) {
                float norm = w_virtual / block_scales_[b];
                uint8_t new_idx;
                switch (fmt) {
                    case NativeFormat::OIL1: {
                        if (norm > 0.5f) new_idx = 2;
                        else if (norm < -0.5f) new_idx = 0;
                        else new_idx = 1;
                        break;
                    }
                    case NativeFormat::OIL4: {
                        new_idx = global_oil4_codebook().nearest(norm);
                        break;
                    }
                    case NativeFormat::OIL8: {
                        new_idx = global_codebook().nearest(norm);
                        break;
                    }
                }
                indices_[i] = new_idx;
            }
        }
    }
}

void NativeOILWeightStore::reallocate_by_sensitivity(const float* sensitivity,
                                                      float frac_oil8,
                                                      float frac_spark) {
    if (num_weights_ == 0) return;
    // Compute per-block max sensitivity (avoids sorting 67M individual weights)
    std::vector<std::pair<float, size_t>> block_sens(num_blocks_);
    for (size_t b = 0; b < num_blocks_; b++) {
        size_t start = b * block_size_;
        size_t end = std::min(num_weights_, start + block_size_);
        float max_s = 0.0f;
        for (size_t i = start; i < end; i++) max_s = std::max(max_s, sensitivity[i]);
        block_sens[b] = {max_s, b};
    }
    // Sort blocks by sensitivity descending
    std::sort(block_sens.begin(), block_sens.end(),
              [](auto& a, auto& b) { return a.first > b.first; });
    
    // Assign formats: top frac_oil8 → OIL8, next frac_spark → OIL1, rest → OIL4
    size_t oil8_blocks = (size_t)(num_blocks_ * frac_oil8);
    size_t spark_blocks = (size_t)(num_blocks_ * frac_spark);
    for (size_t i = 0; i < num_blocks_; i++) {
        size_t b = block_sens[i].second;
        if (i < oil8_blocks)
            formats_[b] = NativeFormat::OIL8;
        else if (i < oil8_blocks + spark_blocks)
            formats_[b] = NativeFormat::OIL1;
        else
            formats_[b] = NativeFormat::OIL4;
    }
}

uint8_t NativeOILWeightStore::get_index(size_t i) const {
    return indices_[i];
}

float NativeOILWeightStore::get_scale(size_t i) const {
    size_t b = i / block_size_;
    return block_scales_[b];
}

NativeFormat NativeOILWeightStore::get_format(size_t i) const {
    size_t b = i / block_size_;
    return formats_[b];
}

float NativeOILWeightStore::get_weight(size_t i) const {
    size_t b = i / block_size_;
    float s = block_scales_[b];
    NativeFormat fmt = formats_[b];
    switch (fmt) {
        case NativeFormat::OIL1: return s * oil1_value(indices_[i]);
        case NativeFormat::OIL4:    return s * global_oil4_codebook().centroid(indices_[i]);
        case NativeFormat::OIL8:    return s * global_codebook().centroid(indices_[i]);
        default: return 0.0f;
    }
}

} // namespace native
} // namespace oil
