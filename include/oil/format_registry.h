#pragma once

#include "oil/types.h"
#include <string>
#include <vector>

namespace oil {

enum class RegFormat : uint32_t {
    OIL1 = 0,
    OIL2,
    OIL4,
    OIL8,
    OIL16,
    OIL32,
    OIL1_GRP,
    OIL2_GRP,
    OIL4_GRP,
    OIL8_GRP,
    OIL16_GRP,
    SPARK_SPARSE,
    SPARK_SPARSE_GRP,
    SPARK_Q0,
    SPARK_Q0_GRP,

    MIX_OIL8_OIL2_01_99,
    MIX_OIL8_OIL4_05_95,
    MIX_OIL4_OIL2_10_90,
    MIX_OIL8_OIL2_10_90,
    MIX_SPARK_OIL8_05_95,
    MIX_OIL16_OIL4_01_99,
    MIX_OIL16_OIL8_05_95,
    MIX_OIL32_OIL8_01_99,

    QUAD_OIL2_OIL4_OIL8_OIL16,
    QUAD_OIL4_OIL8_OIL16_OIL32,
};

struct FormatDescriptor {
    std::string name;
    RegFormat id;
    float bpw;
    int num_centroids;
    bool lossless;
    bool grouped;
    int num_groups;
    float group_size;
    float est_mse;
    std::string description;
};

struct MixDescriptor {
    std::string name;
    RegFormat id;
    int num_tiers;
    RegFormat tier1_fmt;
    float tier1_ratio;
    RegFormat tier2_fmt;
    float tier2_ratio;
    RegFormat tier3_fmt;
    float tier3_ratio;
    RegFormat tier4_fmt;
    float tier4_ratio;
    float effective_bpw;
};

struct QuantResult {
    FormatDescriptor format;
    std::vector<uint8_t> indices;
    std::vector<float> codebook_fp32;
    std::vector<float> group_scales;
    std::vector<float> group_zero_points;
    float global_scale;
    int64_t num_elements;
    int block_size;
    bool success;
};

class FormatRegistry {
public:
    static const std::vector<FormatDescriptor>& get_all_singles();
    static const std::vector<MixDescriptor>& get_all_two_mixes();
    static const std::vector<MixDescriptor>& get_all_four_mixes();

    static FormatDescriptor get_single_format(float target_bpw);
    static MixDescriptor get_two_mix(float target_bpw);
    static MixDescriptor get_four_mix(float target_bpw);

    static QuantResult quantize(const float* data, int64_t n, const FormatDescriptor& fmt);
    static QuantResult quantize_block(const float* block, size_t block_size, const FormatDescriptor& fmt);
    static void dequantize(const QuantResult& qr, float* output, int64_t n);
    static float measure_mse(const float* original, const float* dequantized, int64_t n);

    static float evaluate_format_quality(const float* data, int64_t n, const FormatDescriptor& fmt);
    static float evaluate_format_quality_weighted(const float* data, const float* gradients,
                                                  int64_t n, const FormatDescriptor& fmt,
                                                  float fisher_weight = 0.5f);
    static FormatDescriptor select_best_format(float target_bpw, const float* data, int64_t n);
    static MixDescriptor select_best_mix(float target_bpw, const float* data, int64_t n);
    static std::vector<FormatDescriptor> apply_forced_distribution(
            float target_bpw, int num_formats, const float* data, int64_t n);

    static std::string get_format_table();
    static FormatDescriptor parse_format_name(const std::string& name);

    static float compute_average_bpw(const std::vector<FormatDescriptor>& assignment);

    static QuantResult quantize_oil1(const float* data, int64_t n);
    static QuantResult quantize_oil2(const float* data, int64_t n);
    static QuantResult quantize_oil4(const float* data, int64_t n);
    static QuantResult quantize_oil8(const float* data, int64_t n);
    static QuantResult quantize_spark_q0(const float* data, int64_t n, int block_size);
    static QuantResult quantize_spark_sparse(const float* data, int64_t n);

private:
    static void lloyd_max_train(const float* data, size_t n, float* centroids, int k);
    static int nearest_centroid(float val, const float* centroids, int k);
    static void quantize_normal(const float* data, int64_t n,
                                float* centroids, int k,
                                std::vector<uint8_t>& indices);
    static FormatDescriptor find_closest_single(float target_bpw);
};

inline RegFormat format_to_regformat(Format f) {
    switch (f) {
        case Format::OIL1:      return RegFormat::OIL1;
        case Format::OIL2:      return RegFormat::OIL2;
        case Format::OIL4:      return RegFormat::OIL4;
        case Format::OIL8:      return RegFormat::OIL8;
        case Format::OIL16:     return RegFormat::OIL16;
        case Format::OIL32:     return RegFormat::OIL32;
        case Format::OIL1_GRP:  return RegFormat::OIL1_GRP;
        case Format::OIL2_GRP:  return RegFormat::OIL2_GRP;
        case Format::OIL4_GRP:  return RegFormat::OIL4_GRP;
        case Format::OIL8_GRP:  return RegFormat::OIL8_GRP;
        case Format::OIL16_GRP: return RegFormat::OIL16_GRP;
        case Format::SPARK_SPARSE:      return RegFormat::SPARK_SPARSE;
        case Format::SPARK_SPARSE_GRP:  return RegFormat::SPARK_SPARSE_GRP;
        case Format::SPARK_Q0:          return RegFormat::SPARK_Q0;
        case Format::SPARK_Q0_GRP:      return RegFormat::SPARK_Q0_GRP;
        default: return RegFormat::OIL32;
    }
}

inline Format regformat_to_format(RegFormat rf) {
    switch (rf) {
        case RegFormat::OIL1:      return Format::OIL1;
        case RegFormat::OIL2:      return Format::OIL2;
        case RegFormat::OIL4:      return Format::OIL4;
        case RegFormat::OIL8:      return Format::OIL8;
        case RegFormat::OIL16:     return Format::OIL16;
        case RegFormat::OIL32:     return Format::OIL32;
        case RegFormat::OIL1_GRP:  return Format::OIL1_GRP;
        case RegFormat::OIL2_GRP:  return Format::OIL2_GRP;
        case RegFormat::OIL4_GRP:  return Format::OIL4_GRP;
        case RegFormat::OIL8_GRP:  return Format::OIL8_GRP;
        case RegFormat::OIL16_GRP: return Format::OIL16_GRP;
        case RegFormat::SPARK_SPARSE:      return Format::SPARK_SPARSE;
        case RegFormat::SPARK_SPARSE_GRP:  return Format::SPARK_SPARSE_GRP;
        case RegFormat::SPARK_Q0:          return Format::SPARK_Q0;
        case RegFormat::SPARK_Q0_GRP:      return Format::SPARK_Q0_GRP;
        default: return Format::OIL32;
    }
}

} // namespace oil
