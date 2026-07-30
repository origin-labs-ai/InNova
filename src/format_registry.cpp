#include "oil/format_registry.h"
#include "oil/codebook.h"
#include "oil/math.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <limits>
#include <array>
#include <random>

namespace oil {

static constexpr int OIL_BLOCK_SIZE = 32;

static std::vector<FormatDescriptor> build_singles() {
    std::vector<FormatDescriptor> v;
    v.push_back({"OIL1",          RegFormat::OIL1,          1.0f,  1,    false, false, 1,  1.0f, 8.50e-1f, "block mean (1 centroid per 32 elements)"});
    v.push_back({"OIL2",          RegFormat::OIL2,          2.0f,  4,    false, false, 1024, 4.0f, 2.00e-2f, "Per-block mean-shift + global Lloyd-Max 4 centroids (2-bit)"});
    v.push_back({"OIL4",          RegFormat::OIL4,          4.0f,  16,   false, false, 1024, 16.0f, 2.00e-3f, "Per-block mean-shift + global Lloyd-Max 16 centroids (4-bit)"});
    v.push_back({"OIL8",          RegFormat::OIL8,          8.0f,  256,  false, false, 1024, 256.0f, 5.00e-5f, "Per-block mean-shift + global Lloyd-Max 256 centroids"});
    v.push_back({"OIL16",         RegFormat::OIL16,         16.0f, 0,    false, false, 1,  0.0f, 2.00e-7f, "FP16 storage"});
    v.push_back({"OIL32",         RegFormat::OIL32,         32.0f, 0,    true,  false, 1,  0.0f, 0.0f,    "FP32 identity (lossless)"});
    v.push_back({"OIL1_GRP",      RegFormat::OIL1_GRP,      1.0f,   1,    false, true,  1024, 1024.0f, 7.50e-1f, "1-bit + per-group FP32 scale/zp (group_size=1024)"});
    v.push_back({"OIL2_GRP",      RegFormat::OIL2_GRP,      2.0f,   4,    false, true,  1024, 1024.0f, 9.00e-2f, "Lloyd-Max 4 centroids + per-group FP32 scale/zp (group_size=1024)"});
    v.push_back({"OIL4_GRP",      RegFormat::OIL4_GRP,      4.0f,   16,   false, true,  1024, 1024.0f, 9.00e-3f, "Lloyd-Max 16 centroids + per-group FP32 scale/zp (group_size=1024)"});
    v.push_back({"OIL8_GRP",      RegFormat::OIL8_GRP,      8.0f,   256,  false, true,  1024, 1024.0f, 5.00e-5f, "Lloyd-Max 256 centroids + per-group FP32 scale/zp (group_size=1024)"});
    v.push_back({"OIL16_GRP",     RegFormat::OIL16_GRP,     16.0f,  256,  false, true,  1024, 1024.0f, 3.00e-3f, "256 FP32 centroids + per-group FP32 scale/zp (group_size=1024)"});
    v.push_back({"SPARK_SPARSE",  RegFormat::SPARK_SPARSE,  2.0f,   0,    false, false, 1,    0.0f, 1e-15f,    "Threshold sparsity (uint16 index, int8 value)"});
    v.push_back({"SPARK_SPARSE_GRP", RegFormat::SPARK_SPARSE_GRP, 2.0f, 0, false, true,  1024, 0.0f, 2.20e-4f, "Sparse + per-group int8 scale (group_size=1024)"});
    v.push_back({"SPARK_Q0",      RegFormat::SPARK_Q0,      1.50f,  4,    false, false, 32,   4.0f, 1.70e-1f, "Sign-bit quantized with FP16 scale"});
    v.push_back({"SPARK_Q0_GRP",  RegFormat::SPARK_Q0_GRP,  1.50f,  4,    false, true,  1024, 4.0f, 1.70e-1f, "Sign-bit + per-group scale (group_size=1024)"});
    return v;
}

static std::vector<MixDescriptor> build_two_mixes() {
    std::vector<MixDescriptor> v;
    v.push_back({"OIL8+OIL2_1_99",    RegFormat::MIX_OIL8_OIL2_01_99,    2, RegFormat::OIL8,    0.01f, RegFormat::OIL2,    0.99f, RegFormat::OIL2,   0.0f, RegFormat::OIL2,   0.0f, 2.08f});
    v.push_back({"OIL8+OIL4_5_95",    RegFormat::MIX_OIL8_OIL4_05_95,    2, RegFormat::OIL8,    0.05f, RegFormat::OIL4,    0.95f, RegFormat::OIL4,   0.0f, RegFormat::OIL4,   0.0f, 4.20f});
    v.push_back({"OIL4+OIL2_10_90",   RegFormat::MIX_OIL4_OIL2_10_90,   2, RegFormat::OIL4,    0.10f, RegFormat::OIL2,    0.90f, RegFormat::OIL2,   0.0f, RegFormat::OIL2,   0.0f, 2.30f});
    v.push_back({"OIL8+OIL2_10_90",   RegFormat::MIX_OIL8_OIL2_10_90,   2, RegFormat::OIL8,    0.10f, RegFormat::OIL2,    0.90f, RegFormat::OIL2,   0.0f, RegFormat::OIL2,   0.0f, 2.60f});
    v.push_back({"SPARK+OIL8_5_95",   RegFormat::MIX_SPARK_OIL8_05_95,  2, RegFormat::SPARK_Q0,0.05f, RegFormat::OIL8,    0.95f, RegFormat::OIL8,   0.0f, RegFormat::OIL8,   0.0f, 7.62f});
    v.push_back({"OIL16+OIL4_1_99",   RegFormat::MIX_OIL16_OIL4_01_99,  2, RegFormat::OIL16,   0.01f, RegFormat::OIL4,    0.99f, RegFormat::OIL4,   0.0f, RegFormat::OIL4,   0.0f, 4.16f});
    v.push_back({"OIL16+OIL8_5_95",   RegFormat::MIX_OIL16_OIL8_05_95,  2, RegFormat::OIL16,   0.05f, RegFormat::OIL8,    0.95f, RegFormat::OIL8,   0.0f, RegFormat::OIL8,   0.0f, 8.40f});
    v.push_back({"OIL32+OIL8_1_99",   RegFormat::MIX_OIL32_OIL8_01_99,  2, RegFormat::OIL32,   0.01f, RegFormat::OIL8,    0.99f, RegFormat::OIL8,   0.0f, RegFormat::OIL8,   0.0f, 8.31f});
    return v;
}

static std::vector<MixDescriptor> build_four_mixes() {
    std::vector<MixDescriptor> v;
    v.push_back({"QUAD_OIL2_OIL4_OIL8_OIL16",  RegFormat::QUAD_OIL2_OIL4_OIL8_OIL16,  4,
                 RegFormat::OIL16, 0.01f, RegFormat::OIL8, 0.05f,
                 RegFormat::OIL4, 0.24f, RegFormat::OIL2, 0.70f, 2.78f});
    v.push_back({"QUAD_OIL4_OIL8_OIL16_OIL32", RegFormat::QUAD_OIL4_OIL8_OIL16_OIL32, 4,
                 RegFormat::OIL32, 0.01f, RegFormat::OIL16, 0.05f,
                 RegFormat::OIL8, 0.24f, RegFormat::OIL4, 0.70f, 4.72f});
    return v;
}

const std::vector<FormatDescriptor>& FormatRegistry::get_all_singles() {
    static const std::vector<FormatDescriptor> s = build_singles();
    return s;
}

const std::vector<MixDescriptor>& FormatRegistry::get_all_two_mixes() {
    static const std::vector<MixDescriptor> s = build_two_mixes();
    return s;
}

const std::vector<MixDescriptor>& FormatRegistry::get_all_four_mixes() {
    static const std::vector<MixDescriptor> s = build_four_mixes();
    return s;
}

FormatDescriptor FormatRegistry::get_single_format(float target_bpw) {
    return find_closest_single(target_bpw);
}

MixDescriptor FormatRegistry::get_two_mix(float target_bpw) {
    const auto& mixes = get_all_two_mixes();
    if (mixes.empty()) return {};
    MixDescriptor best = mixes[0];
    float best_dist = std::fabs(mixes[0].effective_bpw - target_bpw);
    for (size_t i = 1; i < mixes.size(); i++) {
        float d = std::fabs(mixes[i].effective_bpw - target_bpw);
        if (d < best_dist) { best_dist = d; best = mixes[i]; }
    }
    return best;
}

MixDescriptor FormatRegistry::get_four_mix(float target_bpw) {
    const auto& mixes = get_all_four_mixes();
    if (mixes.empty()) return {};
    MixDescriptor best = mixes[0];
    float best_dist = std::fabs(mixes[0].effective_bpw - target_bpw);
    for (size_t i = 1; i < mixes.size(); i++) {
        float d = std::fabs(mixes[i].effective_bpw - target_bpw);
        if (d < best_dist) { best_dist = d; best = mixes[i]; }
    }
    return best;
}

FormatDescriptor FormatRegistry::find_closest_single(float target_bpw) {
    const auto& singles = get_all_singles();
    if (singles.empty()) return {};
    FormatDescriptor best = singles[0];
    float best_dist = std::fabs(singles[0].bpw - target_bpw);
    for (size_t i = 1; i < singles.size(); i++) {
        float d = std::fabs(singles[i].bpw - target_bpw);
        if (d < best_dist) { best_dist = d; best = singles[i]; }
    }
    return best;
}

void FormatRegistry::lloyd_max_train(const float* data, size_t n, float* centroids, int k) {
    if (!data || n == 0 || k <= 0) return;
    float mn = data[0], mx = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    if (mn == mx) {
        for (int i = 0; i < k; i++) centroids[i] = mn;
        return;
    }
    // K-means++ style initialization: spread centroids proportional to distance
    centroids[0] = mn + (mx - mn) * 0.5f;
    for (int i = 1; i < k; i++) {
        float pos = mn + (mx - mn) * (static_cast<float>(i) + 0.5f) / static_cast<float>(k);
        // Perturb toward data density
        float sum = 0.0f;
        for (size_t j = 0; j < n; j++) sum += std::fabs(data[j] - pos);
        float wsum = 0.0f;
        for (size_t j = 0; j < n; j++) wsum += data[j] * std::fabs(data[j] - pos);
        centroids[i] = (sum > 1e-10f) ? wsum / sum : pos;
    }
    std::vector<size_t> counts(static_cast<size_t>(k), 0);
    std::vector<float> sums(static_cast<size_t>(k), 0.0f);
    // More iterations for better convergence (50 vs 20)
    for (int iter = 0; iter < 50; iter++) {
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(sums.begin(), sums.end(), 0.0f);
        for (size_t i = 0; i < n; i++) {
            int c = nearest_centroid(data[i], centroids, k);
            counts[static_cast<size_t>(c)]++;
            sums[static_cast<size_t>(c)] += data[i];
        }
        bool converged = true;
        for (int i = 0; i < k; i++) {
            if (counts[static_cast<size_t>(i)] > 0) {
                float new_c = sums[static_cast<size_t>(i)] / static_cast<float>(counts[static_cast<size_t>(i)]);
                if (std::fabs(new_c - centroids[i]) > 1e-8f) converged = false;
                centroids[i] = new_c;
            }
        }
        if (converged) break;
    }
}

int FormatRegistry::nearest_centroid(float val, const float* centroids, int k) {
    int best = 0;
    float best_d = std::fabs(val - centroids[0]);
    for (int i = 1; i < k; i++) {
        float d = std::fabs(val - centroids[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

void FormatRegistry::quantize_normal(const float* data, int64_t n,
                                     float* centroids, int k,
                                     std::vector<uint8_t>& indices) {
    indices.resize(static_cast<size_t>(n));
    // Error feedback: distribute quantization error to nearby weights (GPTQ-style)
    std::vector<float> adjusted(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) adjusted[static_cast<size_t>(i)] = data[i];
    for (int64_t i = 0; i < n; i++) {
        int c = nearest_centroid(adjusted[static_cast<size_t>(i)], centroids, k);
        indices[static_cast<size_t>(i)] = static_cast<uint8_t>(c);
        float error = adjusted[static_cast<size_t>(i)] - centroids[c];
        // Distribute error to next 3 weights (like GPTQ's error diffusion)
        for (int64_t j = i + 1; j < std::min(i + 4, n); j++) {
            adjusted[static_cast<size_t>(j)] += error * 0.25f;
        }
    }
}

QuantResult FormatRegistry::quantize(const float* data, int64_t n,
                                     const FormatDescriptor& fmt) {
    QuantResult qr;
    qr.format = fmt;
    qr.num_elements = n;
    qr.block_size = fmt.group_size > 0 ? static_cast<int>(fmt.group_size) : OIL_BLOCK_SIZE;
    qr.success = false;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) return qr;

    if (fmt.id == RegFormat::OIL32) {
        qr.codebook_fp32.resize(static_cast<size_t>(n));
        std::memcpy(qr.codebook_fp32.data(), data, static_cast<size_t>(n) * sizeof(float));
        qr.indices.resize(static_cast<size_t>(n), 0);
        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::OIL16) {
        qr.codebook_fp32.resize(1);
        qr.codebook_fp32[0] = 0.0f;
        qr.indices.resize(static_cast<size_t>(n) * 2);
        std::vector<uint16_t> fp16(static_cast<size_t>(n));
        oil::math::vec_fp32_to_fp16(fp16.data(), data, static_cast<int>(n));
        for (int64_t i = 0; i < n; i++) {
            qr.indices[static_cast<size_t>(i) * 2]     = static_cast<uint8_t>(fp16[static_cast<size_t>(i)] & 0xFFu);
            qr.indices[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>((fp16[static_cast<size_t>(i)] >> 8) & 0xFFu);
        }
        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::SPARK_Q0) {
        return quantize_spark_q0(data, n, 32);
    }

    if (fmt.id == RegFormat::SPARK_SPARSE) {
        return quantize_spark_sparse(data, n);
    }

    if (fmt.id == RegFormat::OIL1) {
        return quantize_oil1(data, n);
    }

    if (fmt.id == RegFormat::OIL2) {
        return quantize_oil2(data, n);
    }

    if (fmt.id == RegFormat::OIL4) {
        return quantize_oil4(data, n);
    }

    if (fmt.id == RegFormat::OIL8) {
        return quantize_oil8(data, n);
    }

    if (fmt.id == RegFormat::OIL2_GRP) {
        int group_sz = 1024;
        int k = 4;
        int64_t num_groups = (n + group_sz - 1) / group_sz;

        // Normalize all data to [0,1] for codebook training
        float gmin_all = data[0], gmax_all = data[0];
        for (int64_t i = 1; i < n; i++) {
            if (data[i] < gmin_all) gmin_all = data[i];
            if (data[i] > gmax_all) gmax_all = data[i];
        }
        float grange_all = gmax_all - gmin_all;
        if (grange_all < 1e-10f) grange_all = 1.0f;
        std::vector<float> normalized(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++)
            normalized[static_cast<size_t>(i)] = (data[i] - gmin_all) / grange_all;

        std::vector<float> centroids(static_cast<size_t>(k));
        lloyd_max_train(normalized.data(), static_cast<size_t>(n), centroids.data(), k);
        qr.codebook_fp32 = centroids;
        qr.group_scales.resize(static_cast<size_t>(num_groups));
        qr.group_zero_points.resize(static_cast<size_t>(num_groups));

        std::vector<uint8_t> raw(static_cast<size_t>(n));

        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float gmin = data[gstart], gmax = data[gstart];
            for (int64_t i = gstart + 1; i < gend; i++) {
                if (data[i] < gmin) gmin = data[i];
                if (data[i] > gmax) gmax = data[i];
            }
            float range = gmax - gmin;
            if (range < 1e-10f) range = 1.0f;
            qr.group_scales[static_cast<size_t>(g)] = range;
            qr.group_zero_points[static_cast<size_t>(g)] = gmin;

            // Error feedback on normalized values within group
            int64_t gsz = gend - gstart;
            std::vector<float> adjusted(static_cast<size_t>(gsz));
            for (int64_t i = 0; i < gsz; i++)
                adjusted[static_cast<size_t>(i)] = (data[gstart + i] - gmin) / range;

            for (int64_t i = 0; i < gsz; i++) {
                int best = 0;
                float best_d = std::fabs(adjusted[static_cast<size_t>(i)] - centroids[0]);
                for (int c = 1; c < k; c++) {
                    float d = std::fabs(adjusted[static_cast<size_t>(i)] - centroids[c]);
                    if (d < best_d) { best_d = d; best = c; }
                }
                raw[static_cast<size_t>(gstart + i)] = static_cast<uint8_t>(best);
                float err = adjusted[static_cast<size_t>(i)] - centroids[best];
                for (int64_t j = i + 1; j < std::min(i + 4, gsz); j++)
                    adjusted[static_cast<size_t>(j)] += err * 0.25f;
            }
        }

        size_t packed = static_cast<size_t>((n + 3) / 4);
        qr.indices.resize(packed, 0);
        for (int64_t i = 0; i < n; i++) {
            size_t b = static_cast<size_t>(i / 4), o = static_cast<size_t>((i % 4) * 2);
            qr.indices[b] |= (raw[static_cast<size_t>(i)] & 0x3) << static_cast<int>(o);
        }

        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::OIL4_GRP) {
        int group_sz = 1024;
        int k = 16;
        int64_t num_groups = (n + group_sz - 1) / group_sz;

        // Normalize all data to [0,1] for codebook training
        float gmin_all = data[0], gmax_all = data[0];
        for (int64_t i = 1; i < n; i++) {
            if (data[i] < gmin_all) gmin_all = data[i];
            if (data[i] > gmax_all) gmax_all = data[i];
        }
        float grange_all = gmax_all - gmin_all;
        if (grange_all < 1e-10f) grange_all = 1.0f;
        std::vector<float> normalized(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++)
            normalized[static_cast<size_t>(i)] = (data[i] - gmin_all) / grange_all;

        std::vector<float> centroids(static_cast<size_t>(k));
        lloyd_max_train(normalized.data(), static_cast<size_t>(n), centroids.data(), k);
        qr.codebook_fp32 = centroids;
        qr.group_scales.resize(static_cast<size_t>(num_groups));
        qr.group_zero_points.resize(static_cast<size_t>(num_groups));

        std::vector<uint8_t> raw(static_cast<size_t>(n));

        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float gmin = data[gstart], gmax = data[gstart];
            for (int64_t i = gstart + 1; i < gend; i++) {
                if (data[i] < gmin) gmin = data[i];
                if (data[i] > gmax) gmax = data[i];
            }
            float range = gmax - gmin;
            if (range < 1e-10f) range = 1.0f;
            qr.group_scales[static_cast<size_t>(g)] = range;
            qr.group_zero_points[static_cast<size_t>(g)] = gmin;

            int64_t gsz = gend - gstart;
            std::vector<float> adjusted(static_cast<size_t>(gsz));
            for (int64_t i = 0; i < gsz; i++)
                adjusted[static_cast<size_t>(i)] = (data[gstart + i] - gmin) / range;

            for (int64_t i = 0; i < gsz; i++) {
                int best = 0;
                float best_d = std::fabs(adjusted[static_cast<size_t>(i)] - centroids[0]);
                for (int c = 1; c < k; c++) {
                    float d = std::fabs(adjusted[static_cast<size_t>(i)] - centroids[c]);
                    if (d < best_d) { best_d = d; best = c; }
                }
                raw[static_cast<size_t>(gstart + i)] = static_cast<uint8_t>(best);
                float err = adjusted[static_cast<size_t>(i)] - centroids[best];
                for (int64_t j = i + 1; j < std::min(i + 4, gsz); j++)
                    adjusted[static_cast<size_t>(j)] += err * 0.25f;
            }
        }

        size_t packed = static_cast<size_t>((n + 1) / 2);
        qr.indices.resize(packed, 0);
        for (int64_t i = 0; i < n; i++) {
            size_t b = static_cast<size_t>(i / 2), o = static_cast<size_t>((i % 2) * 4);
            qr.indices[b] |= (raw[static_cast<size_t>(i)] & 0xF) << static_cast<int>(o);
        }

        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::OIL8_GRP) {
        int group_sz = 1024;
        int k = 256;
        int64_t num_groups = (n + group_sz - 1) / group_sz;

        // Normalize all data to [0,1] for codebook training
        float gmin_all = data[0], gmax_all = data[0];
        for (int64_t i = 1; i < n; i++) {
            if (data[i] < gmin_all) gmin_all = data[i];
            if (data[i] > gmax_all) gmax_all = data[i];
        }
        float grange_all = gmax_all - gmin_all;
        if (grange_all < 1e-10f) grange_all = 1.0f;
        std::vector<float> normalized(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++)
            normalized[static_cast<size_t>(i)] = (data[i] - gmin_all) / grange_all;

        std::vector<float> centroids(static_cast<size_t>(k));
        lloyd_max_train(normalized.data(), static_cast<size_t>(n), centroids.data(), k);
        qr.codebook_fp32 = centroids;
        qr.group_scales.resize(static_cast<size_t>(num_groups));
        qr.group_zero_points.resize(static_cast<size_t>(num_groups));
        qr.indices.resize(static_cast<size_t>(n));

        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float gmin = data[gstart], gmax = data[gstart];
            for (int64_t i = gstart + 1; i < gend; i++) {
                if (data[i] < gmin) gmin = data[i];
                if (data[i] > gmax) gmax = data[i];
            }
            float range = gmax - gmin;
            if (range < 1e-10f) range = 1.0f;
            qr.group_scales[static_cast<size_t>(g)] = range;
            qr.group_zero_points[static_cast<size_t>(g)] = gmin;

            int64_t gsz = gend - gstart;
            std::vector<float> adjusted(static_cast<size_t>(gsz));
            for (int64_t i = 0; i < gsz; i++)
                adjusted[static_cast<size_t>(i)] = (data[gstart + i] - gmin) / range;

            for (int64_t i = 0; i < gsz; i++) {
                int best = 0;
                float best_d = std::fabs(adjusted[static_cast<size_t>(i)] - centroids[0]);
                for (int c = 1; c < k; c++) {
                    float d = std::fabs(adjusted[static_cast<size_t>(i)] - centroids[c]);
                    if (d < best_d) { best_d = d; best = c; }
                }
                qr.indices[static_cast<size_t>(gstart + i)] = static_cast<uint8_t>(best);
                float err = adjusted[static_cast<size_t>(i)] - centroids[best];
                for (int64_t j = i + 1; j < std::min(i + 4, gsz); j++)
                    adjusted[static_cast<size_t>(j)] += err * 0.25f;
            }
        }

        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::OIL1_GRP) {
        int group_sz = 1024;
        int64_t num_groups = (n + group_sz - 1) / group_sz;
        qr.codebook_fp32.resize(1);
        qr.codebook_fp32[0] = 0.0f;
        qr.group_scales.resize(static_cast<size_t>(num_groups));
        qr.group_zero_points.resize(static_cast<size_t>(num_groups));
        qr.indices.resize(static_cast<size_t>(n));
        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float gmin = data[gstart], gmax = data[gstart];
            for (int64_t i = gstart + 1; i < gend; i++) {
                if (data[i] < gmin) gmin = data[i];
                if (data[i] > gmax) gmax = data[i];
            }
            float range = gmax - gmin;
            if (range < 1e-10f) range = 1.0f;
            qr.group_scales[static_cast<size_t>(g)] = range;
            qr.group_zero_points[static_cast<size_t>(g)] = gmin;
            float mean = (gmin + gmax) * 0.5f;
            for (int64_t i = gstart; i < gend; i++) {
                qr.indices[static_cast<size_t>(i)] = (data[i] >= mean) ? 1 : 0;
            }
        }
        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::OIL16_GRP) {
        int group_sz = 1024;
        int k = 256;
        int64_t num_groups = (n + group_sz - 1) / group_sz;

        // Normalize all data to [0,1] for codebook training
        float gmin_all = data[0], gmax_all = data[0];
        for (int64_t i = 1; i < n; i++) {
            if (data[i] < gmin_all) gmin_all = data[i];
            if (data[i] > gmax_all) gmax_all = data[i];
        }
        float grange_all = gmax_all - gmin_all;
        if (grange_all < 1e-10f) grange_all = 1.0f;
        std::vector<float> normalized(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++)
            normalized[static_cast<size_t>(i)] = (data[i] - gmin_all) / grange_all;

        std::vector<float> centroids(static_cast<size_t>(k));
        lloyd_max_train(normalized.data(), static_cast<size_t>(n), centroids.data(), k);
        qr.codebook_fp32 = centroids;

        qr.group_scales.resize(static_cast<size_t>(num_groups));
        qr.group_zero_points.resize(static_cast<size_t>(num_groups));
        qr.indices.resize(static_cast<size_t>(n));

        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;

            float gmin = data[gstart], gmax = data[gstart];
            for (int64_t i = gstart + 1; i < gend; i++) {
                if (data[i] < gmin) gmin = data[i];
                if (data[i] > gmax) gmax = data[i];
            }
            float range = gmax - gmin;
            if (range < 1e-10f) range = 1.0f;
            qr.group_scales[static_cast<size_t>(g)] = range;
            qr.group_zero_points[static_cast<size_t>(g)] = gmin;

            int64_t gsz = gend - gstart;
            std::vector<float> adjusted(static_cast<size_t>(gsz));
            for (int64_t i = 0; i < gsz; i++)
                adjusted[static_cast<size_t>(i)] = (data[gstart + i] - gmin) / range;

            for (int64_t i = 0; i < gsz; i++) {
                int best = 0;
                float best_d = std::fabs(adjusted[static_cast<size_t>(i)] - centroids[0]);
                for (int c = 1; c < k; c++) {
                    float d = std::fabs(adjusted[static_cast<size_t>(i)] - centroids[c]);
                    if (d < best_d) { best_d = d; best = c; }
                }
                qr.indices[static_cast<size_t>(gstart + i)] = static_cast<uint8_t>(best);
                float err = adjusted[static_cast<size_t>(i)] - centroids[best];
                for (int64_t j = i + 1; j < std::min(i + 4, gsz); j++)
                    adjusted[static_cast<size_t>(j)] += err * 0.25f;
            }
        }

        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::SPARK_Q0_GRP) {
        int group_sz = fmt.group_size > 0 ? static_cast<int>(fmt.group_size) : 32;
        int64_t num_groups = (n + group_sz - 1) / group_sz;

        qr.codebook_fp32.resize(4);
        qr.group_scales.resize(static_cast<size_t>(num_groups));
        qr.group_zero_points.resize(static_cast<size_t>(num_groups));
        qr.indices.resize(static_cast<size_t>(n));

        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;

            float max_abs = 0.0f;
            for (int64_t i = gstart; i < gend; i++) {
                float a = std::fabs(data[i]);
                if (a > max_abs) max_abs = a;
            }
            float scale = max_abs > 0.0f ? max_abs : 1.0f;
            qr.group_scales[static_cast<size_t>(g)] = scale;
            qr.group_zero_points[static_cast<size_t>(g)] = 0.0f;

            for (int64_t i = gstart; i < gend; i++) {
                float normalized = data[i] / scale;
                if (normalized >= 0.5f) qr.indices[static_cast<size_t>(i)] = 3;
                else if (normalized >= 0.0f) qr.indices[static_cast<size_t>(i)] = 2;
                else if (normalized >= -0.5f) qr.indices[static_cast<size_t>(i)] = 1;
                else qr.indices[static_cast<size_t>(i)] = 0;
            }
        }

        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::SPARK_SPARSE_GRP) {
        int group_sz = 1024;
        int64_t num_groups = (n + group_sz - 1) / group_sz;

        qr.group_scales.resize(static_cast<size_t>(num_groups));
        qr.group_zero_points.resize(static_cast<size_t>(num_groups));
        std::vector<uint32_t> sparse_indices;
        std::vector<float> sparse_values;
        std::vector<int> sparse_groups;

        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;

            float gmax = 0.0f;
            for (int64_t i = gstart; i < gend; i++) {
                float a = std::fabs(data[i]);
                if (a > gmax) gmax = a;
            }
            float group_scale = gmax > 0.0f ? gmax : 1.0f;
            float group_threshold = group_scale * 0.01f;
            qr.group_scales[static_cast<size_t>(g)] = group_scale;
            qr.group_zero_points[static_cast<size_t>(g)] = 0.0f;

            for (int64_t i = gstart; i < gend; i++) {
                if (std::fabs(data[i]) > group_threshold) {
                    sparse_indices.push_back(static_cast<uint32_t>(i));
                    sparse_values.push_back(data[i]);
                    sparse_groups.push_back(static_cast<int>(g));
                }
            }
        }

        int64_t nnz = static_cast<int64_t>(sparse_indices.size());
        if (nnz == 0) {
            qr.codebook_fp32.resize(1);
            qr.codebook_fp32[0] = 0.0f;
            qr.indices.resize(4, 0);
            qr.success = true;
            return qr;
        }

        qr.codebook_fp32.resize(1);
        qr.codebook_fp32[0] = 0.0f;
        size_t total_bytes = static_cast<size_t>(4 + nnz * 6);
        qr.indices.resize(total_bytes);
        qr.indices[0] = static_cast<uint8_t>(static_cast<uint32_t>(nnz) & 0xFF);
        qr.indices[1] = static_cast<uint8_t>((static_cast<uint32_t>(nnz) >> 8) & 0xFF);
        qr.indices[2] = static_cast<uint8_t>((static_cast<uint32_t>(nnz) >> 16) & 0xFF);
        qr.indices[3] = static_cast<uint8_t>((static_cast<uint32_t>(nnz) >> 24) & 0xFF);

        for (int64_t i = 0; i < nnz; i++) {
            size_t base = static_cast<size_t>(4 + i * 6);
            uint32_t idx = sparse_indices[static_cast<size_t>(i)];
            int group = sparse_groups[static_cast<size_t>(i)];
            int8_t q_val = static_cast<int8_t>(static_cast<int>(sparse_values[static_cast<size_t>(i)] / qr.group_scales[static_cast<size_t>(group)] * 127.0f));
            qr.indices[base]     = static_cast<uint8_t>(idx & 0xFF);
            qr.indices[base + 1] = static_cast<uint8_t>((idx >> 8) & 0xFF);
            qr.indices[base + 2] = static_cast<uint8_t>((idx >> 16) & 0xFF);
            qr.indices[base + 3] = static_cast<uint8_t>((idx >> 24) & 0xFF);
            qr.indices[base + 4] = static_cast<uint8_t>(static_cast<int>(q_val) + 128);
            qr.indices[base + 5] = static_cast<uint8_t>(group);
        }

        qr.success = true;
        return qr;
    }

    return qr;
}

QuantResult FormatRegistry::quantize_block(const float* block, size_t block_size,
                                           const FormatDescriptor& fmt) {
    return quantize(block, static_cast<int64_t>(block_size), fmt);
}

QuantResult FormatRegistry::quantize_oil1(const float* data, int64_t n) {
    QuantResult qr;
    qr.format = {"OIL1", RegFormat::OIL1, 1.0f, 1, false, false, 1, 1.0f, 0.0f, "block mean (1 centroid per 32 elements)"};
    qr.num_elements = n;
    qr.block_size = OIL_BLOCK_SIZE;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) { qr.success = false; return qr; }

    int64_t num_blocks = (n + OIL_BLOCK_SIZE - 1) / OIL_BLOCK_SIZE;
    qr.codebook_fp32.resize(static_cast<size_t>(num_blocks));
    qr.indices.clear();

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t bstart = b * OIL_BLOCK_SIZE;
        int64_t bend = (bstart + OIL_BLOCK_SIZE <= n) ? bstart + OIL_BLOCK_SIZE : n;
        float sum = 0.0f;
        for (int64_t i = bstart; i < bend; i++) sum += data[i];
        float mean = sum / static_cast<float>(bend - bstart);
        qr.codebook_fp32[static_cast<size_t>(b)] = mean;
    }
    qr.success = true;
    return qr;
}

QuantResult FormatRegistry::quantize_oil2(const float* data, int64_t n) {
    QuantResult qr;
    qr.format = {"OIL2", RegFormat::OIL2, 2.0f, 4, false, false, 1, 4.0f, 0.0f, "Per-block mean-shift + global Lloyd-Max 4 centroids (2-bit packed)"};
    qr.num_elements = n;
    qr.block_size = 1024;
    qr.global_scale = 0.0f;
    qr.success = false;
    if (!data || n <= 0) return qr;
    const int k = 4;
    const int64_t bsz = 1024;
    const int64_t nb = (n + bsz - 1) / bsz;
    std::vector<float> residuals(static_cast<size_t>(n));
    qr.group_zero_points.resize(static_cast<size_t>(nb));
    for (int64_t b = 0; b < nb; b++) {
        int64_t s = b * bsz, e = std::min(s + bsz, n);
        double mu = 0.0;
        for (int64_t i = s; i < e; i++) mu += data[i];
        mu /= static_cast<double>(e - s);
        qr.group_zero_points[static_cast<size_t>(b)] = static_cast<float>(mu);
        for (int64_t i = s; i < e; i++)
            residuals[static_cast<size_t>(i)] = data[i] - static_cast<float>(mu);
    }
    std::vector<float> centroids(static_cast<size_t>(k));
    lloyd_max_train(residuals.data(), static_cast<size_t>(n), centroids.data(), k);
    qr.codebook_fp32 = centroids;
    std::vector<uint8_t> raw(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++)
        raw[static_cast<size_t>(i)] = static_cast<uint8_t>(nearest_centroid(residuals[static_cast<size_t>(i)], centroids.data(), k));
    size_t packed = static_cast<size_t>((n + 3) / 4);
    qr.indices.resize(packed, 0);
    for (int64_t i = 0; i < n; i++) {
        size_t b = static_cast<size_t>(i / 4), o = static_cast<size_t>((i % 4) * 2);
        qr.indices[b] |= (raw[static_cast<size_t>(i)] & 0x3) << static_cast<int>(o);
    }
    qr.success = true;
    return qr;
}

QuantResult FormatRegistry::quantize_oil4(const float* data, int64_t n) {
    QuantResult qr;
    qr.format = {"OIL4", RegFormat::OIL4, 4.0f, 16, false, false, 1, 16.0f, 0.0f, "Per-block mean-shift + global Lloyd-Max 16 centroids (4-bit packed)"};
    qr.num_elements = n;
    qr.block_size = 1024;
    qr.global_scale = 0.0f;
    qr.success = false;
    if (!data || n <= 0) return qr;
    const int k = 16;
    const int64_t bsz = 1024;
    const int64_t nb = (n + bsz - 1) / bsz;
    std::vector<float> residuals(static_cast<size_t>(n));
    qr.group_zero_points.resize(static_cast<size_t>(nb));
    for (int64_t b = 0; b < nb; b++) {
        int64_t s = b * bsz, e = std::min(s + bsz, n);
        double mu = 0.0;
        for (int64_t i = s; i < e; i++) mu += data[i];
        mu /= static_cast<double>(e - s);
        qr.group_zero_points[static_cast<size_t>(b)] = static_cast<float>(mu);
        for (int64_t i = s; i < e; i++)
            residuals[static_cast<size_t>(i)] = data[i] - static_cast<float>(mu);
    }
    std::vector<float> centroids(static_cast<size_t>(k));
    lloyd_max_train(residuals.data(), static_cast<size_t>(n), centroids.data(), k);
    qr.codebook_fp32 = centroids;
    std::vector<uint8_t> raw(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++)
        raw[static_cast<size_t>(i)] = static_cast<uint8_t>(nearest_centroid(residuals[static_cast<size_t>(i)], centroids.data(), k));
    size_t packed = static_cast<size_t>((n + 1) / 2);
    qr.indices.resize(packed, 0);
    for (int64_t i = 0; i < n; i++) {
        size_t b = static_cast<size_t>(i / 2), o = static_cast<size_t>((i % 2) * 4);
        qr.indices[b] |= (raw[static_cast<size_t>(i)] & 0xF) << static_cast<int>(o);
    }
    qr.success = true;
    return qr;
}

QuantResult FormatRegistry::quantize_oil8(const float* data, int64_t n) {
    QuantResult qr;
    qr.format = {"OIL8", RegFormat::OIL8, 8.0f, 256, false, false, 1, 256.0f, 0.0f, "Per-block mean-shift + global Lloyd-Max 256 centroids"};
    qr.num_elements = n;
    qr.block_size = 1024;
    qr.global_scale = 0.0f;
    qr.success = false;
    if (!data || n <= 0) return qr;
    const int k = 256;
    const int64_t bsz = 1024;
    const int64_t nb = (n + bsz - 1) / bsz;
    std::vector<float> residuals(static_cast<size_t>(n));
    qr.group_zero_points.resize(static_cast<size_t>(nb));
    for (int64_t b = 0; b < nb; b++) {
        int64_t s = b * bsz, e = std::min(s + bsz, n);
        double mu = 0.0;
        for (int64_t i = s; i < e; i++) mu += data[i];
        mu /= static_cast<double>(e - s);
        qr.group_zero_points[static_cast<size_t>(b)] = static_cast<float>(mu);
        for (int64_t i = s; i < e; i++)
            residuals[static_cast<size_t>(i)] = data[i] - static_cast<float>(mu);
    }
    std::vector<float> centroids(static_cast<size_t>(k));
    lloyd_max_train(residuals.data(), static_cast<size_t>(n), centroids.data(), k);
    qr.codebook_fp32 = centroids;
    qr.indices.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++)
        qr.indices[static_cast<size_t>(i)] = static_cast<uint8_t>(nearest_centroid(residuals[static_cast<size_t>(i)], centroids.data(), k));
    qr.success = true;
    return qr;
}

QuantResult FormatRegistry::quantize_spark_q0(const float* data, int64_t n, int block_size) {
    QuantResult qr;
    qr.format = {"SPARK_Q0", RegFormat::SPARK_Q0, 1.50f, 4, false, false, block_size, 4.0f, 0.0f, "Sign-bit quantized with FP16 scale"};
    qr.num_elements = n;
    qr.block_size = block_size;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) { qr.success = false; return qr; }

    int64_t num_blocks = (n + block_size - 1) / block_size;
    qr.codebook_fp32.resize(static_cast<size_t>(num_blocks));
    qr.indices.resize(static_cast<size_t>(n), 0);

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t bstart = b * block_size;
        int64_t bend = (bstart + block_size <= n) ? bstart + block_size : n;

        float max_abs = 0.0f;
        for (int64_t i = bstart; i < bend; i++) {
            float a = std::fabs(data[i]);
            if (a > max_abs) max_abs = a;
        }
        float scale = max_abs > 0.0f ? max_abs : 1.0f;
        qr.codebook_fp32[static_cast<size_t>(b)] = scale;

        for (int64_t i = bstart; i < bend; i++) {
            float normalized = data[i] / scale;
            if (normalized >= 0.5f) qr.indices[static_cast<size_t>(i)] = 3;
            else if (normalized >= 0.0f) qr.indices[static_cast<size_t>(i)] = 2;
            else if (normalized >= -0.5f) qr.indices[static_cast<size_t>(i)] = 1;
            else qr.indices[static_cast<size_t>(i)] = 0;
        }
    }
    qr.success = true;
    return qr;
}

QuantResult FormatRegistry::quantize_spark_sparse(const float* data, int64_t n) {
    QuantResult qr;
    qr.format = {"SPARK_SPARSE", RegFormat::SPARK_SPARSE, 2.0f, 0, false, false, 1, 0.0f, 0.0f, "Threshold sparsity (uint16 index, int8 value)"};
    qr.num_elements = n;
    qr.block_size = 1;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) { qr.success = false; return qr; }

    std::vector<uint32_t> sparse_indices;
    std::vector<float> sparse_values;

    float p90_threshold = 0.0f;
    {
        std::vector<float> sorted_abs(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) sorted_abs[static_cast<size_t>(i)] = std::fabs(data[i]);
        std::sort(sorted_abs.begin(), sorted_abs.end());
        size_t p90_idx = static_cast<size_t>(static_cast<double>(n) * 0.90);
        if (p90_idx >= sorted_abs.size()) p90_idx = sorted_abs.size() - 1;
        p90_threshold = sorted_abs[p90_idx];
        if (p90_threshold < 1e-10f) p90_threshold = 1e-4f;
    }

    for (int64_t i = 0; i < n; i++) {
        if (std::fabs(data[i]) > p90_threshold) {
            sparse_indices.push_back(static_cast<uint32_t>(i));
            sparse_values.push_back(data[i]);
        }
    }

    int64_t nnz = static_cast<int64_t>(sparse_indices.size());

    if (nnz == 0) {
        qr.codebook_fp32.resize(1);
        qr.codebook_fp32[0] = 0.0f;
        qr.indices.resize(4, 0);
        qr.success = true;
        return qr;
    }

    float max_abs = 0.0f;
    for (int64_t i = 0; i < nnz; i++) {
        float a = std::fabs(sparse_values[static_cast<size_t>(i)]);
        if (a > max_abs) max_abs = a;
    }
    float global_scale = max_abs > 0.0f ? max_abs : 1.0f;
    qr.codebook_fp32.resize(1);
    qr.codebook_fp32[0] = global_scale;

    size_t total_bytes = static_cast<size_t>(4 + nnz * 6);
    qr.indices.resize(total_bytes);
    qr.indices[0] = static_cast<uint8_t>(static_cast<uint32_t>(nnz) & 0xFF);
    qr.indices[1] = static_cast<uint8_t>((static_cast<uint32_t>(nnz) >> 8) & 0xFF);
    qr.indices[2] = static_cast<uint8_t>((static_cast<uint32_t>(nnz) >> 16) & 0xFF);
    qr.indices[3] = static_cast<uint8_t>((static_cast<uint32_t>(nnz) >> 24) & 0xFF);

    for (int64_t i = 0; i < nnz; i++) {
        size_t base = static_cast<size_t>(4 + i * 6);
        uint32_t idx = sparse_indices[static_cast<size_t>(i)];
        int8_t q_val = static_cast<int8_t>(static_cast<int>(sparse_values[static_cast<size_t>(i)] / global_scale * 127.0f));
        qr.indices[base]     = static_cast<uint8_t>(idx & 0xFF);
        qr.indices[base + 1] = static_cast<uint8_t>((idx >> 8) & 0xFF);
        qr.indices[base + 2] = static_cast<uint8_t>((idx >> 16) & 0xFF);
        qr.indices[base + 3] = static_cast<uint8_t>((idx >> 24) & 0xFF);
        qr.indices[base + 4] = static_cast<uint8_t>(static_cast<int>(q_val) + 128);
        qr.indices[base + 5] = 0;
    }

    qr.success = true;
    return qr;
}

void FormatRegistry::dequantize(const QuantResult& qr, float* output, int64_t n) {
    if (!output || n <= 0 || !qr.success) return;

    if (qr.format.id == RegFormat::OIL32) {
        std::memcpy(output, qr.codebook_fp32.data(), static_cast<size_t>(n) * sizeof(float));
        return;
    }

    if (qr.format.id == RegFormat::OIL16) {
        std::vector<uint16_t> fp16(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            fp16[static_cast<size_t>(i)] = static_cast<uint16_t>(qr.indices[static_cast<size_t>(i) * 2]) |
                                           (static_cast<uint16_t>(qr.indices[static_cast<size_t>(i) * 2 + 1]) << 8);
        }
        oil::math::vec_fp16_to_fp32(output, fp16.data(), static_cast<int>(n));
        return;
    }

    if (qr.format.id == RegFormat::OIL1) {
        int64_t num_blocks = (n + OIL_BLOCK_SIZE - 1) / OIL_BLOCK_SIZE;
        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t bstart = b * OIL_BLOCK_SIZE;
            int64_t bend = (bstart + OIL_BLOCK_SIZE <= n) ? bstart + OIL_BLOCK_SIZE : n;
            float val = (b < static_cast<int64_t>(qr.codebook_fp32.size())) ? qr.codebook_fp32[static_cast<size_t>(b)] : 0.0f;
            for (int64_t i = bstart; i < bend; i++) {
                output[i] = val;
            }
        }
        return;
    }

    if (qr.format.id == RegFormat::SPARK_Q0 || qr.format.id == RegFormat::SPARK_Q0_GRP) {
        int bs = qr.block_size > 0 ? qr.block_size : 32;
        int64_t num_blocks = (n + bs - 1) / bs;
        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t bstart = b * bs;
            int64_t bend = (bstart + bs <= n) ? bstart + bs : n;
            float scale = (b < static_cast<int64_t>(qr.codebook_fp32.size())) ? qr.codebook_fp32[static_cast<size_t>(b)] : 1.0f;
            for (int64_t i = bstart; i < bend; i++) {
                uint8_t idx = qr.indices[static_cast<size_t>(i)];
                float normalized;
                switch (idx) {
                    case 3: normalized = 0.75f; break;
                    case 2: normalized = 0.25f; break;
                    case 1: normalized = -0.25f; break;
                    default: normalized = -0.75f; break;
                }
                output[i] = normalized * scale;
            }
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL2) {
        if (qr.codebook_fp32.empty()) { std::fill(output, output + n, 0.0f); return; }
        int64_t bsz = qr.block_size > 0 ? qr.block_size : 1024;
        for (int64_t i = 0; i < n; i++) {
            int64_t blk = i / bsz;
            size_t packed_idx = static_cast<size_t>(i / 4), off = static_cast<size_t>((i % 4) * 2);
            int idx = (packed_idx < qr.indices.size()) ? (qr.indices[packed_idx] >> static_cast<int>(off)) & 0x3 : 0;
            float centroid = (static_cast<size_t>(idx) < qr.codebook_fp32.size()) ? qr.codebook_fp32[static_cast<size_t>(idx)] : 0.0f;
            float mean = (static_cast<size_t>(blk) < qr.group_zero_points.size()) ? qr.group_zero_points[static_cast<size_t>(blk)] : 0.0f;
            output[i] = centroid + mean;
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL4) {
        if (qr.codebook_fp32.empty()) { std::fill(output, output + n, 0.0f); return; }
        int64_t bsz = qr.block_size > 0 ? qr.block_size : 1024;
        for (int64_t i = 0; i < n; i++) {
            int64_t blk = i / bsz;
            size_t packed_idx = static_cast<size_t>(i / 2), off = static_cast<size_t>((i % 2) * 4);
            int idx = (packed_idx < qr.indices.size()) ? (qr.indices[packed_idx] >> static_cast<int>(off)) & 0xF : 0;
            float centroid = (static_cast<size_t>(idx) < qr.codebook_fp32.size()) ? qr.codebook_fp32[static_cast<size_t>(idx)] : 0.0f;
            float mean = (static_cast<size_t>(blk) < qr.group_zero_points.size()) ? qr.group_zero_points[static_cast<size_t>(blk)] : 0.0f;
            output[i] = centroid + mean;
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL8) {
        if (qr.codebook_fp32.empty()) { std::fill(output, output + n, 0.0f); return; }
        int64_t bsz = qr.block_size > 0 ? qr.block_size : 1024;
        for (int64_t i = 0; i < n; i++) {
            int64_t blk = i / bsz;
            size_t idx = static_cast<size_t>(qr.indices[static_cast<size_t>(i)]);
            float centroid = (idx < qr.codebook_fp32.size()) ? qr.codebook_fp32[idx] : 0.0f;
            float mean = (static_cast<size_t>(blk) < qr.group_zero_points.size()) ? qr.group_zero_points[static_cast<size_t>(blk)] : 0.0f;
            output[i] = centroid + mean;
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL2_GRP) {
        if (qr.codebook_fp32.empty()) { std::fill(output, output + n, 0.0f); return; }
        int group_sz = 1024;
        for (int64_t g = 0; g < (n + group_sz - 1) / group_sz; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float scale = (g < static_cast<int64_t>(qr.group_scales.size())) ? qr.group_scales[static_cast<size_t>(g)] : 1.0f;
            float zp = (g < static_cast<int64_t>(qr.group_zero_points.size())) ? qr.group_zero_points[static_cast<size_t>(g)] : 0.0f;
            for (int64_t i = gstart; i < gend; i++) {
                size_t b = static_cast<size_t>(i / 4), o = static_cast<size_t>((i % 4) * 2);
                int idx = (b < qr.indices.size()) ? (qr.indices[b] >> static_cast<int>(o)) & 0x3 : 0;
                float centroid = (static_cast<size_t>(idx) < qr.codebook_fp32.size()) ? qr.codebook_fp32[static_cast<size_t>(idx)] : 0.0f;
                output[i] = zp + centroid * scale;
            }
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL4_GRP) {
        if (qr.codebook_fp32.empty()) { std::fill(output, output + n, 0.0f); return; }
        int group_sz = 1024;
        for (int64_t g = 0; g < (n + group_sz - 1) / group_sz; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float scale = (g < static_cast<int64_t>(qr.group_scales.size())) ? qr.group_scales[static_cast<size_t>(g)] : 1.0f;
            float zp = (g < static_cast<int64_t>(qr.group_zero_points.size())) ? qr.group_zero_points[static_cast<size_t>(g)] : 0.0f;
            for (int64_t i = gstart; i < gend; i++) {
                size_t b = static_cast<size_t>(i / 2), o = static_cast<size_t>((i % 2) * 4);
                int idx = (b < qr.indices.size()) ? (qr.indices[b] >> static_cast<int>(o)) & 0xF : 0;
                float centroid = (static_cast<size_t>(idx) < qr.codebook_fp32.size()) ? qr.codebook_fp32[static_cast<size_t>(idx)] : 0.0f;
                output[i] = zp + centroid * scale;
            }
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL8_GRP) {
        if (qr.codebook_fp32.empty()) { std::fill(output, output + n, 0.0f); return; }
        int group_sz = 1024;
        for (int64_t g = 0; g < (n + group_sz - 1) / group_sz; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float scale = (g < static_cast<int64_t>(qr.group_scales.size())) ? qr.group_scales[static_cast<size_t>(g)] : 1.0f;
            float zp = (g < static_cast<int64_t>(qr.group_zero_points.size())) ? qr.group_zero_points[static_cast<size_t>(g)] : 0.0f;
            for (int64_t i = gstart; i < gend; i++) {
                int idx = (static_cast<size_t>(i) < qr.indices.size()) ? static_cast<int>(qr.indices[static_cast<size_t>(i)]) : 0;
                float centroid = (static_cast<size_t>(idx) < qr.codebook_fp32.size()) ? qr.codebook_fp32[static_cast<size_t>(idx)] : 0.0f;
                output[i] = zp + centroid * scale;
            }
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL1_GRP) {
        int group_sz = 1024;
        int64_t num_groups = (n + group_sz - 1) / group_sz;
        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float scale = (g < static_cast<int64_t>(qr.group_scales.size())) ? qr.group_scales[static_cast<size_t>(g)] : 1.0f;
            float zp = (g < static_cast<int64_t>(qr.group_zero_points.size())) ? qr.group_zero_points[static_cast<size_t>(g)] : 0.0f;
            float midpoint = zp + scale * 0.5f;
            for (int64_t i = gstart; i < gend; i++) {
                int bit = (static_cast<size_t>(i) < qr.indices.size()) ? static_cast<int>(qr.indices[static_cast<size_t>(i)]) : 0;
                output[i] = zp + (bit ? scale : 0.0f);
            }
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL16_GRP) {
        if (qr.codebook_fp32.empty()) { std::fill(output, output + n, 0.0f); return; }
        int group_sz = 1024;
        for (int64_t g = 0; g < (n + group_sz - 1) / group_sz; g++) {
            int64_t gstart = g * group_sz;
            int64_t gend = (gstart + group_sz <= n) ? gstart + group_sz : n;
            float scale = (g < static_cast<int64_t>(qr.group_scales.size())) ? qr.group_scales[static_cast<size_t>(g)] : 1.0f;
            float zp = (g < static_cast<int64_t>(qr.group_zero_points.size())) ? qr.group_zero_points[static_cast<size_t>(g)] : 0.0f;
            for (int64_t i = gstart; i < gend; i++) {
                int idx = (static_cast<size_t>(i) < qr.indices.size()) ? static_cast<int>(qr.indices[static_cast<size_t>(i)]) : 0;
                float centroid = (static_cast<size_t>(idx) < qr.codebook_fp32.size()) ? qr.codebook_fp32[static_cast<size_t>(idx)] : 0.0f;
                output[i] = zp + centroid * scale;
            }
        }
        return;
    }

    if (qr.format.id == RegFormat::SPARK_SPARSE || qr.format.id == RegFormat::SPARK_SPARSE_GRP) {
        std::fill(output, output + n, 0.0f);
        if (qr.indices.size() < 4) return;
        uint32_t nnz = static_cast<uint32_t>(qr.indices[0]) |
                      (static_cast<uint32_t>(qr.indices[1]) << 8) |
                      (static_cast<uint32_t>(qr.indices[2]) << 16) |
                      (static_cast<uint32_t>(qr.indices[3]) << 24);
        float gs = 1.0f;
        if (qr.format.id == RegFormat::SPARK_SPARSE && !qr.codebook_fp32.empty())
            gs = qr.codebook_fp32[0];
        for (uint32_t i = 0; i < nnz; i++) {
            size_t base = static_cast<size_t>(4 + i * 6);
            if (base + 4 >= qr.indices.size()) break;
            uint32_t idx = static_cast<uint32_t>(qr.indices[base]) |
                          (static_cast<uint32_t>(qr.indices[base + 1]) << 8) |
                          (static_cast<uint32_t>(qr.indices[base + 2]) << 16) |
                          (static_cast<uint32_t>(qr.indices[base + 3]) << 24);
            int8_t q_val = static_cast<int8_t>(static_cast<int>(qr.indices[base + 4]) - 128);
            float scale = gs;
            if (qr.format.id == RegFormat::SPARK_SPARSE_GRP) {
                int group = static_cast<int>(qr.indices[base + 5]);
                scale = (static_cast<size_t>(group) < qr.group_scales.size()) ? qr.group_scales[static_cast<size_t>(group)] : 1.0f;
            }
            float val = static_cast<float>(q_val) / 127.0f * scale;
            if (idx < static_cast<uint32_t>(n)) output[static_cast<size_t>(idx)] = val;
        }
        return;
    }

    std::fill(output, output + n, 0.0f);
}

float FormatRegistry::measure_mse(const float* original, const float* dequantized, int64_t n) {
    if (!original || !dequantized || n <= 0) return 0.0f;
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double diff = static_cast<double>(original[i]) - static_cast<double>(dequantized[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum / static_cast<double>(n));
}

float FormatRegistry::evaluate_format_quality(const float* data, int64_t n,
                                              const FormatDescriptor& fmt) {
    if (!data || n <= 0) return 0.0f;
    QuantResult qr = quantize(data, n, fmt);
    if (!qr.success) return 0.0f;
    std::vector<float> dequant(static_cast<size_t>(n));
    dequantize(qr, dequant.data(), n);
    return measure_mse(data, dequant.data(), n);
}

float FormatRegistry::evaluate_format_quality_weighted(const float* data, const float* gradients,
                                                       int64_t n, const FormatDescriptor& fmt,
                                                       float fisher_weight) {
    if (!data || !gradients || n <= 0) return 0.0f;

    QuantResult qr = quantize(data, n, fmt);
    if (!qr.success) return 0.0f;
    std::vector<float> dequant(static_cast<size_t>(n));
    dequantize(qr, dequant.data(), n);

    double sum = 0.0, weight_sum = 0.0;
    float max_grad = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float g = std::fabs(gradients[i]);
        if (g > max_grad) max_grad = g;
    }
    if (max_grad == 0.0f) max_grad = 1.0f;
    for (int64_t i = 0; i < n; i++) {
        float importance = (1.0f - fisher_weight) + fisher_weight * (std::fabs(gradients[i]) / max_grad);
        double diff = static_cast<double>(data[i]) - static_cast<double>(dequant[static_cast<size_t>(i)]);
        sum += importance * diff * diff;
        weight_sum += importance;
    }
    return static_cast<float>(sum / weight_sum);
}

FormatDescriptor FormatRegistry::select_best_format(float target_bpw, const float* data, int64_t n) {
    const auto& singles = get_all_singles();
    FormatDescriptor best;
    float best_mse = 1e30f;
    for (const auto& fmt : singles) {
        if (std::fabs(fmt.bpw - target_bpw) > 0.5f) continue;
        float mse = evaluate_format_quality(data, n, fmt);
        if (mse < best_mse) { best_mse = mse; best = fmt; }
    }
    if (!best.name.empty()) return best;
    return find_closest_single(target_bpw);
}

MixDescriptor FormatRegistry::select_best_mix(float target_bpw, const float* data, int64_t n) {
    (void)data; (void)n;
    const auto& two = get_all_two_mixes();
    const auto& four = get_all_four_mixes();
    MixDescriptor best_two = get_two_mix(target_bpw);
    MixDescriptor best_four = get_four_mix(target_bpw);
    float d2 = std::fabs(best_two.effective_bpw - target_bpw);
    float d4 = std::fabs(best_four.effective_bpw - target_bpw);
    if (d2 <= d4) return best_two;
    return best_four;
}

std::vector<FormatDescriptor> FormatRegistry::apply_forced_distribution(
        float target_bpw, int num_formats, const float* data, int64_t n) {
    std::vector<FormatDescriptor> result;
    if (num_formats <= 0) return result;
    const auto& singles = get_all_singles();
    std::vector<std::pair<float, int>> candidates;
    for (int i = 0; i < static_cast<int>(singles.size()); i++) {
        float d = std::fabs(singles[static_cast<size_t>(i)].bpw - target_bpw);
        candidates.push_back({d, i});
    }
    std::sort(candidates.begin(), candidates.end());
    int count = std::min(num_formats, static_cast<int>(candidates.size()));
    for (int i = 0; i < count; i++) {
        result.push_back(singles[static_cast<size_t>(candidates[static_cast<size_t>(i)].second)]);
    }
    return result;
}

float FormatRegistry::compute_average_bpw(const std::vector<FormatDescriptor>& assignment) {
    if (assignment.empty()) return 0.0f;
    double sum = 0.0;
    for (const auto& f : assignment) sum += f.bpw;
    return static_cast<float>(sum / static_cast<double>(assignment.size()));
}

std::string FormatRegistry::get_format_table() {
    const auto& singles = get_all_singles();
    std::ostringstream os;
    os << "Format          BPW   Centroids  Lossless  Grouped  GrpSize  MSE        Description\n";
    os << "--------------  ----  ---------  --------  -------  -------  ---------- ----------------\n";
    for (const auto& s : singles) {
        os << s.name;
        int pad = 16 - static_cast<int>(s.name.size());
        if (pad > 0) os << std::string(static_cast<size_t>(pad), ' ');
        os.width(4); os << std::right << s.bpw << "   ";
        os.width(9); os << std::right << s.num_centroids << "  ";
        os << (s.lossless ? "Yes" : "No ") << "      ";
        os << (s.grouped ? "Yes" : "No ") << "     ";
        os.width(7); os << std::right << s.group_size << "  ";
        os.width(10); os << std::right << s.est_mse << "  ";
        os << s.description << "\n";
    }
    return os.str();
}

FormatDescriptor FormatRegistry::parse_format_name(const std::string& name) {
    const auto& singles = get_all_singles();
    for (const auto& s : singles) {
        if (s.name == name) return s;
    }
    return {};
}

} // namespace oil