#include "oil/format_registry.h"
#include "oil/codebook.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <limits>
#include <array>
#include <random>

namespace oil {

static constexpr int OIL_BLOCK_SIZE = 32;
static constexpr int DEFAULT_GRP_GROUP_SIZE = 128;

static std::vector<FormatDescriptor> build_singles() {
    std::vector<FormatDescriptor> v;
    v.push_back({"OIL1",          RegFormat::OIL1,          1.0f,  1,    false, false, 1,  1.0f, 0.0f,   "Lloyd-Max 1 centroid per block (lossy)"});
    v.push_back({"OIL2",          RegFormat::OIL2,          2.0f,  4,    false, false, 1,  4.0f, 1.18e-1f, "Lloyd-Max 4 centroids (lossy)"});
    v.push_back({"OIL4",          RegFormat::OIL4,          4.0f,  16,   false, false, 1,  16.0f, 9.40e-3f, "Lloyd-Max 16 centroids (lossy)"});
    v.push_back({"OIL8",          RegFormat::OIL8,          8.0f,  256,  false, false, 1,  256.0f, 3.12e-5f, "Lloyd-Max 256 centroids (lossy)"});
    v.push_back({"OIL16",         RegFormat::OIL16,         16.0f, 0,    false, false, 1,  0.0f, 0.0f,    "FP16 storage (near-lossless)"});
    v.push_back({"OIL32",         RegFormat::OIL32,         32.0f, 0,    true,  false, 1,  0.0f, 0.0f,    "FP32 identity (lossless)"});
    v.push_back({"OIL1_GRP",      RegFormat::OIL1_GRP,      1.0f,  1,    true,  true,  128, 1.0f, 0.0f,   "OIL1 + per-group scale/zp (lossless)"});
    v.push_back({"OIL2_GRP",      RegFormat::OIL2_GRP,      2.0f,  4,    true,  true,  128, 4.0f, 0.0f,   "OIL2 + per-group scale/zp (lossless)"});
    v.push_back({"OIL4_GRP",      RegFormat::OIL4_GRP,      4.0f,  16,   true,  true,  128, 16.0f, 0.0f,  "OIL4 + per-group scale/zp (lossless)"});
    v.push_back({"OIL8_GRP",      RegFormat::OIL8_GRP,      8.0f,  256,  true,  true,  128, 256.0f, 0.0f,  "OIL8 + per-group scale/zp (lossless)"});
    v.push_back({"OIL16_GRP",     RegFormat::OIL16_GRP,     16.0f, 0,    true,  true,  128, 0.0f, 0.0f,   "FP16 + per-group scale/zp (lossless)"});
    v.push_back({"SPARK_SPARSE",  RegFormat::SPARK_SPARSE,  1.5f,  0,    false, false, 1,  0.0f, 1.85e-5f, "Threshold sparsity (index,value pairs)"});
    v.push_back({"SPARK_SPARSE_GRP", RegFormat::SPARK_SPARSE_GRP, 2.0f, 0, true,  true,  128, 0.0f, 0.0f, "Sparse + per-group scale (lossless)"});
    v.push_back({"SPARK_Q0",      RegFormat::SPARK_Q0,      1.5f,  4,    false, false, 32, 4.0f, 1.86e-5f, "Sign bit + shared FP16 scale (lossy)"});
    v.push_back({"SPARK_Q0_GRP",  RegFormat::SPARK_Q0_GRP,  1.5f,  4,    true,  true,  32, 4.0f, 0.0f,    "Sign + per-group scale (lossless)"});
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
    for (int i = 0; i < k; i++) {
        centroids[i] = mn + (mx - mn) * (i + 0.5f) / k;
    }
    std::vector<size_t> counts(k, 0);
    std::vector<float> sums(k, 0.0f);
    for (int iter = 0; iter < 20; iter++) {
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(sums.begin(), sums.end(), 0.0f);
        for (size_t i = 0; i < n; i++) {
            int c = nearest_centroid(data[i], centroids, k);
            counts[c]++;
            sums[c] += data[i];
        }
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) centroids[i] = sums[i] / counts[i];
        }
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
    indices.resize((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        indices[(size_t)i] = (uint8_t)nearest_centroid(data[i], centroids, k);
    }
}

void FormatRegistry::quantize_grp(const float* data, int64_t n,
                                  float* centroids, int k, int group_size,
                                  std::vector<uint8_t>& indices,
                                  std::vector<float>& group_scales,
                                  std::vector<float>& group_zero_points) {
    indices.resize((size_t)n);
    int64_t num_groups = (n + group_size - 1) / group_size;
    group_scales.resize((size_t)num_groups);
    group_zero_points.resize((size_t)num_groups);

    for (int64_t g = 0; g < num_groups; g++) {
        int64_t gstart = g * group_size;
        int64_t gend = (gstart + group_size <= n) ? gstart + group_size : n;
        int64_t gn = gend - gstart;

        float gmin = data[gstart];
        float gmax = data[gstart];
        for (int64_t i = gstart + 1; i < gend; i++) {
            if (data[i] < gmin) gmin = data[i];
            if (data[i] > gmax) gmax = data[i];
        }

        float scale = (gmax > gmin) ? (gmax - gmin) : 1.0f;
        float zp = gmin;
        group_scales[(size_t)g] = scale;
        group_zero_points[(size_t)g] = zp;

        std::vector<float> local(k);
        for (int c = 0; c < k; c++) {
            local[c] = zp + scale * (c + 0.5f) / k;
        }

        for (int64_t i = gstart; i < gend; i++) {
            int best_c = 0;
            float best_d = std::fabs(data[i] - local[0]);
            for (int c = 1; c < k; c++) {
                float d = std::fabs(data[i] - local[c]);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            indices[(size_t)i] = (uint8_t)best_c;
        }
    }

    for (int c = 0; c < k; c++) centroids[c] = (float)c;
}

QuantResult FormatRegistry::quantize(const float* data, int64_t n,
                                     const FormatDescriptor& fmt) {
    QuantResult qr;
    qr.format = fmt;
    qr.num_elements = n;
    qr.block_size = fmt.group_size > 0 ? (int)fmt.group_size : OIL_BLOCK_SIZE;
    qr.success = false;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) return qr;

    if (fmt.id == RegFormat::OIL32) {
        qr.codebook_fp32.resize((size_t)n);
        std::memcpy(qr.codebook_fp32.data(), data, (size_t)n * sizeof(float));
        qr.indices.resize((size_t)n, 0);
        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::OIL16) {
        qr.codebook_fp32.resize(1);
        qr.codebook_fp32[0] = 0.0f;
        qr.indices.resize((size_t)n * 2);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(data);
        for (int64_t i = 0; i < n; i++) {
            qr.indices[(size_t)i * 2]     = (uint8_t)(src[i] & 0xFF);
            qr.indices[(size_t)i * 2 + 1] = (uint8_t)((src[i] >> 8) & 0xFF);
        }
        qr.success = true;
        return qr;
    }

    if (fmt.id == RegFormat::SPARK_Q0) {
        return quantize_spark_q0(data, n, 32);
    }

    if (fmt.id == RegFormat::SPARK_SPARSE) {
        return quantize_spark_sparse(data, n, 1e-4f);
    }

    if (fmt.id == RegFormat::OIL1) {
        return quantize_oil1(data, n);
    }

    if (fmt.id == RegFormat::SPARK_Q0_GRP) {
        QuantResult inner = quantize_spark_q0(data, n, 32);
        inner.format = fmt;
        inner.format.id = RegFormat::SPARK_Q0_GRP;
        inner.format.lossless = true;
        inner.success = true;
        return inner;
    }

    if (fmt.id == RegFormat::SPARK_SPARSE_GRP) {
        QuantResult inner = quantize_spark_sparse(data, n, 0.0f);
        inner.format = fmt;
        inner.format.id = RegFormat::SPARK_SPARSE_GRP;
        inner.format.lossless = true;
        inner.success = true;
        return inner;
    }

    if (fmt.grouped) {
        int k = fmt.num_centroids > 0 ? fmt.num_centroids : 16;
        int group_sz = fmt.group_size > 0 ? fmt.group_size : DEFAULT_GRP_GROUP_SIZE;

        qr.codebook_fp32.resize((size_t)k);
        quantize_grp(data, n, qr.codebook_fp32.data(), k, group_sz,
                     qr.indices, qr.group_scales, qr.group_zero_points);
        qr.success = true;
        return qr;
    }

    if (fmt.num_centroids > 0 && fmt.num_centroids <= 256) {
        int k = fmt.num_centroids;
        qr.codebook_fp32.resize((size_t)k);
        lloyd_max_train(data, (size_t)n, qr.codebook_fp32.data(), k);
        quantize_normal(data, n, qr.codebook_fp32.data(), k, qr.indices);
        qr.success = true;
        return qr;
    }

    return qr;
}

QuantResult FormatRegistry::quantize_block(const float* block, size_t block_size,
                                           const FormatDescriptor& fmt) {
    return quantize(block, (int64_t)block_size, fmt);
}

QuantResult FormatRegistry::quantize_oil1(const float* data, int64_t n) {
    QuantResult qr;
    qr.format = {"OIL1", RegFormat::OIL1, 1.0f, 1, false, false, 1, 1.0f, 0.0f, "Lloyd-Max 1 centroid per block (lossy)"};
    qr.num_elements = n;
    qr.block_size = OIL_BLOCK_SIZE;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) { qr.success = false; return qr; }

    int64_t num_blocks = (n + OIL_BLOCK_SIZE - 1) / OIL_BLOCK_SIZE;
    qr.codebook_fp32.resize((size_t)num_blocks);
    qr.indices.resize((size_t)n, 0);

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t bstart = b * OIL_BLOCK_SIZE;
        int64_t bend = (bstart + OIL_BLOCK_SIZE <= n) ? bstart + OIL_BLOCK_SIZE : n;
        float sum = 0.0f;
        for (int64_t i = bstart; i < bend; i++) sum += data[i];
        float mean = sum / (float)(bend - bstart);
        qr.codebook_fp32[(size_t)b] = mean;
        for (int64_t i = bstart; i < bend; i++) {
            qr.indices[(size_t)i] = 0;
        }
    }
    qr.success = true;
    return qr;
}

QuantResult FormatRegistry::quantize_spark_q0(const float* data, int64_t n, int block_size) {
    QuantResult qr;
    qr.format = {"SPARK_Q0", RegFormat::SPARK_Q0, 1.5f, 4, false, false, block_size, 4.0f, 0.0f, "Sign + shared FP16 scale (lossy)"};
    qr.num_elements = n;
    qr.block_size = block_size;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) { qr.success = false; return qr; }

    int64_t num_blocks = (n + block_size - 1) / block_size;
    qr.codebook_fp32.resize((size_t)num_blocks);
    qr.indices.resize((size_t)n, 0);

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t bstart = b * block_size;
        int64_t bend = (bstart + block_size <= n) ? bstart + block_size : n;

        float max_abs = 0.0f;
        for (int64_t i = bstart; i < bend; i++) {
            float a = std::fabs(data[i]);
            if (a > max_abs) max_abs = a;
        }
        float scale = max_abs > 0.0f ? max_abs : 1.0f;
        qr.codebook_fp32[(size_t)b] = scale;

        for (int64_t i = bstart; i < bend; i++) {
            float normalized = data[i] / scale;
            if (normalized >= 0.5f) qr.indices[(size_t)i] = 3;
            else if (normalized >= 0.0f) qr.indices[(size_t)i] = 2;
            else if (normalized >= -0.5f) qr.indices[(size_t)i] = 1;
            else qr.indices[(size_t)i] = 0;
        }
    }
    qr.success = true;
    return qr;
}

QuantResult FormatRegistry::quantize_spark_sparse(const float* data, int64_t n, float threshold) {
    QuantResult qr;
    qr.format = {"SPARK_SPARSE", RegFormat::SPARK_SPARSE, 1.5f, 0, false, false, 1, 0.0f, 0.0f, "Threshold sparsity (index,value pairs)"};
    qr.num_elements = n;
    qr.block_size = 1;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) { qr.success = false; return qr; }

    std::vector<float> indices_f;
    std::vector<float> values_f;

    for (int64_t i = 0; i < n; i++) {
        if (std::fabs(data[i]) > threshold) {
            indices_f.push_back((float)i);
            values_f.push_back(data[i]);
        }
    }

    int64_t nnz = (int64_t)indices_f.size();
    qr.codebook_fp32.resize((size_t)(nnz * 2 + 1));
    qr.codebook_fp32[0] = (float)nnz;
    for (int64_t i = 0; i < nnz; i++) {
        qr.codebook_fp32[(size_t)(1 + i)] = indices_f[(size_t)i];
        qr.codebook_fp32[(size_t)(1 + nnz + i)] = values_f[(size_t)i];
    }

    qr.indices.resize((size_t)n, 0);
    qr.success = true;
    return qr;
}

void FormatRegistry::dequantize(const QuantResult& qr, float* output, int64_t n) {
    if (!output || n <= 0 || !qr.success) return;

    if (qr.format.id == RegFormat::OIL32) {
        std::memcpy(output, qr.codebook_fp32.data(), (size_t)n * sizeof(float));
        return;
    }

    if (qr.format.id == RegFormat::OIL16) {
        for (int64_t i = 0; i < n; i++) {
            uint16_t h = (uint16_t)qr.indices[(size_t)i * 2] |
                         ((uint16_t)qr.indices[(size_t)i * 2 + 1] << 8);
            float val;
            std::memcpy(&val, &h, sizeof(uint16_t));
            const uint16_t* hp = reinterpret_cast<const uint16_t*>(&h);
            uint32_t Sign = (uint32_t)(hp[0] >> 15) << 31;
            uint32_t Mantissa = ((uint32_t)(hp[0] & 0x3FF)) << 13;
            uint32_t Exponent = ((uint32_t)((hp[0] >> 10) & 0x1F) + 127 - 15) << 23;
            uint32_t Bits = Sign | Exponent | Mantissa;
            std::memcpy(output + i, &Bits, sizeof(float));
        }
        return;
    }

    if (qr.format.id == RegFormat::OIL1) {
        int64_t num_blocks = (n + OIL_BLOCK_SIZE - 1) / OIL_BLOCK_SIZE;
        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t bstart = b * OIL_BLOCK_SIZE;
            int64_t bend = (bstart + OIL_BLOCK_SIZE <= n) ? bstart + OIL_BLOCK_SIZE : n;
            float val = (b < (int64_t)qr.codebook_fp32.size()) ? qr.codebook_fp32[(size_t)b] : 0.0f;
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
            float scale = (b < (int64_t)qr.codebook_fp32.size()) ? qr.codebook_fp32[(size_t)b] : 1.0f;
            for (int64_t i = bstart; i < bend; i++) {
                uint8_t idx = qr.indices[(size_t)i];
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

    if (qr.format.id == RegFormat::SPARK_SPARSE || qr.format.id == RegFormat::SPARK_SPARSE_GRP) {
        std::fill(output, output + n, 0.0f);
        if (!qr.codebook_fp32.empty()) {
            int64_t nnz = (int64_t)qr.codebook_fp32[0];
            for (int64_t i = 0; i < nnz && (1 + i) < (int64_t)qr.codebook_fp32.size(); i++) {
                int64_t idx = (int64_t)qr.codebook_fp32[(size_t)(1 + i)];
                float val = qr.codebook_fp32[(size_t)(1 + nnz + i)];
                if (idx >= 0 && idx < n) output[idx] = val;
            }
        }
        return;
    }

    if (qr.format.grouped && !qr.group_scales.empty()) {
        int k = qr.format.num_centroids > 0 ? qr.format.num_centroids : 16;
        int gs = qr.format.group_size > 0 ? qr.format.group_size : DEFAULT_GRP_GROUP_SIZE;
        int64_t num_groups = (n + gs - 1) / gs;
        for (int64_t g = 0; g < num_groups; g++) {
            int64_t gstart = g * gs;
            int64_t gend = (gstart + gs <= n) ? gstart + gs : n;
            float scale = (g < (int64_t)qr.group_scales.size()) ? qr.group_scales[(size_t)g] : 1.0f;
            float zp = (g < (int64_t)qr.group_zero_points.size()) ? qr.group_zero_points[(size_t)g] : 0.0f;
            for (int64_t i = gstart; i < gend; i++) {
                int idx = qr.indices[(size_t)i] % k;
                output[i] = zp + scale * ((float)idx + 0.5f) / (float)k;
            }
        }
        return;
    }

    if (qr.format.num_centroids > 0 && !qr.codebook_fp32.empty() && !qr.format.grouped) {
        int k = (int)qr.codebook_fp32.size();
        for (int64_t i = 0; i < n; i++) {
            int idx = qr.indices[(size_t)i] % k;
            output[i] = qr.codebook_fp32[idx];
        }
        return;
    }

    for (int64_t i = 0; i < n; i++) {
        output[i] = 0.0f;
    }
}

float FormatRegistry::measure_mse(const float* original, const float* dequantized, int64_t n) {
    if (!original || !dequantized || n <= 0) return 0.0f;
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double diff = (double)original[i] - (double)dequantized[i];
        sum += diff * diff;
    }
    return (float)(sum / n);
}

float FormatRegistry::evaluate_format_quality(const float* data, int64_t n,
                                              const FormatDescriptor& fmt) {
    if (!data || n <= 0) return 0.0f;
    QuantResult qr = quantize(data, n, fmt);
    if (!qr.success) return 0.0f;
    std::vector<float> dequant((size_t)n);
    dequantize(qr, dequant.data(), n);
    return measure_mse(data, dequant.data(), n);
}

float FormatRegistry::evaluate_format_quality_weighted(const float* data, const float* gradients,
                                                       int64_t n, const FormatDescriptor& fmt,
                                                       float fisher_weight) {
    if (!data || !gradients || n <= 0) return 0.0f;
    std::vector<float> weighted((size_t)n);
    float max_grad = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float g = std::fabs(gradients[i]);
        if (g > max_grad) max_grad = g;
    }
    if (max_grad == 0.0f) max_grad = 1.0f;
    for (int64_t i = 0; i < n; i++) {
        float importance = (1.0f - fisher_weight) + fisher_weight * (std::fabs(gradients[i]) / max_grad);
        weighted[(size_t)i] = data[i] * importance;
    }
    QuantResult qr = quantize(weighted.data(), n, fmt);
    if (!qr.success) return 0.0f;
    std::vector<float> dequant((size_t)n);
    dequantize(qr, dequant.data(), n);
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double diff = (double)weighted[(size_t)i] - (double)dequant[(size_t)i];
        sum += diff * diff;
    }
    return (float)(sum / n);
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
    for (int i = 0; i < (int)singles.size(); i++) {
        float d = std::fabs(singles[i].bpw - target_bpw);
        candidates.push_back({d, i});
    }
    std::sort(candidates.begin(), candidates.end());
    int count = std::min(num_formats, (int)candidates.size());
    for (int i = 0; i < count; i++) {
        result.push_back(singles[candidates[i].second]);
    }
    return result;
}

float FormatRegistry::compute_average_bpw(const std::vector<FormatDescriptor>& assignment) {
    if (assignment.empty()) return 0.0f;
    double sum = 0.0;
    for (const auto& f : assignment) sum += f.bpw;
    return (float)(sum / assignment.size());
}

std::string FormatRegistry::get_format_table() {
    const auto& singles = get_all_singles();
    std::ostringstream os;
    os << "Format          BPW   Centroids  Lossless  Grouped  GrpSize  MSE        Description\n";
    os << "--------------  ----  ---------  --------  -------  -------  ---------- ----------------\n";
    for (const auto& s : singles) {
        os << s.name;
        int pad = 16 - (int)s.name.size();
        if (pad > 0) os << std::string(pad, ' ');
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
