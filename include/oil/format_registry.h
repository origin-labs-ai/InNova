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

    // SPARK_MIX — adaptive, hard-capped mixes. Exactly the claimed BPW,
    // never a bit more: the per-block byte budget is enforced by the
    // adaptive allocator (allocate_mix_blocks). BPW values below are the
    // HONEST weighted sums of the member formats' true stored BPW.
    MIX_SPARK_Q0,      // 1.925 BPW exact — TWI_MIX: OIL8_GRP (1%) + OIL4_GRP (2%) + OIL2_GRP (52%) + OIL1_GRP (45%)
    QUAD_SPARK_Q1,     // 2.075 BPW exact — QUAD_MIX: OIL32 (1%) + OIL8_GRP (1%) + OIL2_GRP (46%) + OIL1_GRP (52%)
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
    // Adaptive mixes (SPARK_MIX): blocks are assigned to member formats by
    // measured reconstruction benefit per byte, under a HARD budget equal to
    // the claimed effective_bpw. Non-adaptive mixes keep the registry ratios.
    bool adaptive = false;
};

struct QuantResult {
    FormatDescriptor format;
    // Canonical wire payload: the exact bytes the block codec stores on disk,
    // concatenated per 256-weight wire block (block_codec.h). dequantize()
    // walks block_idx_bytes/block_cb_bytes to slice each block's payload.
    std::vector<uint8_t> indices;
    std::vector<uint8_t> codebook;         // wire codebook channel (OIL1 block means)
    std::vector<uint32_t> block_idx_bytes; // per wire block: indices payload bytes
    std::vector<uint32_t> block_cb_bytes;  // per wire block: codebook payload bytes
    // In-memory metadata kept for compatibility (OIL32 FP32 copy and any
    // caller-visible scales). NOT stored on disk — the wire payload in
    // indices/codebook is the complete serialized form.
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
    // Per-block layout produced by the adaptive mix allocator. Block sizes
    // may differ per block (row-aligned segmentation for narrow 2D tensors);
    // every block's byte claim plus the format table entry is fully
    // self-describing on disk (num_weights is stored per block).
    struct MixBlockPlan {
        std::vector<Format> formats;       // per block, wire formats 0..14
        std::vector<int64_t> block_starts; // flat offset of each block in data
        std::vector<int64_t> block_lens;   // weights per block
    };

    static const std::vector<FormatDescriptor>& get_all_singles();
    static const std::vector<MixDescriptor>& get_all_twi_mixes();
    static const std::vector<MixDescriptor>& get_all_four_mixes();

    static FormatDescriptor get_single_format(float target_bpw);
    static MixDescriptor get_twi_mix(float target_bpw);
    static MixDescriptor get_four_mix(float target_bpw);

    static QuantResult quantize(const float* data, int64_t n, const FormatDescriptor& fmt);
    static QuantResult quantize_block(const float* block, size_t block_size, const FormatDescriptor& fmt);
    static void dequantize(const QuantResult& qr, float* output, int64_t n);
    // Counts every byte required to reconstruct the quantized payload.
    static size_t serialized_size_bytes(const QuantResult& qr);
    static float actual_bpw(const QuantResult& qr);
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

    // Adaptive mix allocation (SPARK_MIX_Q0 / SPARK_MIX_Q1).
    // Per-block member-format assignment that treats the claimed effective
    // BPW as a HARD byte budget: the returned plan NEVER costs more than
    // ceil(effective_bpw * n / 8) bytes, and spends every byte only where it
    // measurably reduces reconstruction error (benefit per byte spent).
    // `shape` (optional): for rank-2 tensors whose column count is below the
    // block size, blocks are aligned to whole ROWS (one block per row) so
    // scales and tiers follow the matrix's row/column structure; otherwise
    // fixed block_size blocks are used, which are row-aligned whenever the
    // block size divides the column count.
    static MixBlockPlan allocate_mix_blocks(const MixDescriptor& mix,
                                            const float* data, int64_t n,
                                            int block_size,
                                            const std::vector<int64_t>* shape = nullptr);

    static QuantResult quantize_oil1(const float* data, int64_t n);
    static QuantResult quantize_oil2(const float* data, int64_t n);
    static QuantResult quantize_oil4(const float* data, int64_t n);
    static QuantResult quantize_oil8(const float* data, int64_t n);
    static QuantResult quantize_spark_q0(const float* data, int64_t n, int block_size);
    static QuantResult quantize_spark_sparse(const float* data, int64_t n);

private:
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
