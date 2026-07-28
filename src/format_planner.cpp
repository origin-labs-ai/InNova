#include "oil/format_planner.h"
#include "oil/format_registry.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>

namespace oil {

FormatPlanner::FormatPlanner(float t) : target_bpw_(t) {}

void FormatPlanner::score_importance(const Tensor& weights,
                                      const Tensor& calibration_activations,
                                      int block_size, ImportanceMetric metric) {
    if (metric == ImportanceMetric::FISHER_DIAG) {
        score_importance_fisher(weights, calibration_activations, block_size);
        return;
    }
    const float* wd = (const float*)weights.data();
    const float* ad = (const float*)calibration_activations.data();
    int64_t n = weights.numel();
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;
    importance_scores_.resize((size_t)num_blocks);

    for (int64_t b = 0; b < num_blocks; b++) {
        double score = 0;
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        for (int64_t i = start; i < end; i++) {
            float w = wd[i];
            float a = ad ? std::fabs(ad[i]) : 1.0f;
            score += (double)std::fabs(w) * (double)a;
        }
        importance_scores_[b] = (float)(score / (double)(end - start));
    }
}

void FormatPlanner::score_importance_fisher(const Tensor& weights,
                                             const Tensor& gradients,
                                             int block_size) {
    const float* wd = (const float*)weights.data();
    const float* gd = (const float*)gradients.data();
    int64_t n = weights.numel();
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;
    importance_scores_.resize((size_t)num_blocks);

    for (int64_t b = 0; b < num_blocks; b++) {
        double score = 0;
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        for (int64_t i = start; i < end; i++) {
            float g = gd[i];
            score += (double)(g * g);
        }
        score = std::sqrt(score / (double)(end - start));
        score *= (double)std::fabs(wd[start]);
        importance_scores_[b] = (float)score;
    }
}

void FormatPlanner::compute_format_mix(int num_blocks, float target_bpw,
                                        int& oil32, int& oil16, int& oil8,
                                        int& oil4, int& oil2, int& spark, int& oil1) {
    const float bpw_oil32 = 32.0f;
    const float bpw_oil16 = 16.0f;
    const float bpw_oil8 = 8.0f;
    const float bpw_oil4 = 4.0f;
    const float bpw_oil2 = 2.0f;
    const float bpw_spark = 2.0f;

    float f32 = 0, f16 = 0, f8 = 0, f4 = 0, f2 = 0, f_sp = 0, f1 = 0;

    if (target_bpw >= bpw_oil32) {
        f32 = 1.0f;
    } else if (target_bpw >= bpw_oil16) {
        f32 = (target_bpw - bpw_oil16) / (bpw_oil32 - bpw_oil16);
        f16 = 1.0f - f32;
    } else if (target_bpw >= bpw_oil8) {
        f16 = (target_bpw - bpw_oil8) / (bpw_oil16 - bpw_oil8);
        f8 = 1.0f - f16;
    } else if (target_bpw >= bpw_oil4) {
        f8 = (target_bpw - bpw_oil4) / (bpw_oil8 - bpw_oil4);
        f4 = 1.0f - f8;
    } else if (target_bpw >= bpw_oil2) {
        f4 = (target_bpw - bpw_oil2) / (bpw_oil4 - bpw_oil2);
        f2 = 1.0f - f4;
    } else if (target_bpw >= 1.0f) {
        // 1.0-2.0: linear OIL1 (1.0 BPW) to SPARK (2.0 BPW, lowest quality)
        f_sp = (target_bpw - 1.0f) / (bpw_oil2 - 1.0f);
        f1 = 1.0f - f_sp;
    } else {
        f1 = 1.0f;
    }

    oil32 = (int)std::round(f32 * (double)num_blocks);
    oil16 = (int)std::round(f16 * (double)num_blocks);
    oil8 = (int)std::round(f8 * (double)num_blocks);
    oil4 = (int)std::round(f4 * (double)num_blocks);
    oil2 = (int)std::round(f2 * (double)num_blocks);
    spark = (int)std::round(f_sp * (double)num_blocks);
    oil1 = (int)std::round(f1 * (double)num_blocks);

    // Normalize to exact block count
    int total = oil32 + oil16 + oil8 + oil4 + oil2 + spark + oil1;
    int diff = num_blocks - total;
    if (diff > 0) oil1 += diff;
    else if (diff < 0) {
        if (oil1 >= -diff) oil1 += diff;
        else { oil1 = 0; spark = (std::max)(0, spark + diff); }
    }

    oil32 = (std::min)(oil32, num_blocks);
    oil16 = (std::min)(oil16, num_blocks - oil32);
    oil8 = (std::min)(oil8, num_blocks - oil32 - oil16);
    oil4 = (std::min)(oil4, num_blocks - oil32 - oil16 - oil8);
    oil2 = (std::min)(oil2, num_blocks - oil32 - oil16 - oil8 - oil4);
    spark = (std::min)(spark, num_blocks - oil32 - oil16 - oil8 - oil4 - oil2);
    oil1 = num_blocks - oil32 - oil16 - oil8 - oil4 - oil2 - spark;
    if (oil1 < 0) oil1 = 0;
}

static RegFormat bpw_to_reg_format(float bpw) {
    if (bpw >= 31.99f) return RegFormat::OIL32;
    if (bpw >= 15.99f) return RegFormat::OIL16;
    if (bpw >= 7.99f)  return RegFormat::OIL8;
    if (bpw >= 3.99f)  return RegFormat::OIL4;
    if (bpw >= 1.99f)  return RegFormat::OIL2;
    return RegFormat::OIL1;
}

FormatPlan FormatPlanner::allocate(int num_weight_blocks, int weights_per_block) {
    FormatPlan plan;
    plan.target_bpw = target_bpw_;
    plan.blocks.resize(num_weight_blocks);
    plan.num_oil32_blocks = 0;
    plan.num_oil16_blocks = 0;
    plan.num_oil8_blocks = 0;
    plan.num_oil4_blocks = 0;
    plan.num_oil2_blocks = 0;
    plan.num_spark_blocks = 0;
    plan.num_oil1_blocks = 0;
    plan.uses_mix = false;

    plan.selected_single = FormatRegistry::get_single_format(target_bpw_);

    const auto& two_mixes = FormatRegistry::get_all_two_mixes();
    float best_two_diff = 1e9f;
    for (const auto& m : two_mixes) {
        float diff = std::fabs(m.effective_bpw - target_bpw_);
        if (diff < best_two_diff) {
            best_two_diff = diff;
            plan.selected_mix = m;
        }
    }

    const auto& four_mixes = FormatRegistry::get_all_four_mixes();
    float best_four_diff = 1e9f;
    MixDescriptor best_four;
    for (const auto& m : four_mixes) {
        float diff = std::fabs(m.effective_bpw - target_bpw_);
        if (diff < best_four_diff) {
            best_four_diff = diff;
            best_four = m;
        }
    }

    float single_diff = std::fabs(plan.selected_single.bpw - target_bpw_);

    if (best_two_diff < single_diff && best_two_diff < best_four_diff) {
        plan.uses_mix = true;

        int crit_count = (int)std::round(plan.selected_mix.tier1_ratio * num_weight_blocks);
        int rest_count = num_weight_blocks - crit_count;

        std::vector<int> indices(num_weight_blocks);
        for (int i = 0; i < num_weight_blocks; i++) indices[i] = i;
        if (!importance_scores_.empty()) {
            std::sort(indices.begin(), indices.end(), [this](int a, int b) {
                float sa = a < (int)importance_scores_.size() ? importance_scores_[a] : 0;
                float sb = b < (int)importance_scores_.size() ? importance_scores_[b] : 0;
                return sa > sb;
            });
        }

        RegFormat crit_fmt = plan.selected_mix.tier1_fmt;
        RegFormat rest_fmt = plan.selected_mix.tier2_fmt;

        for (int i = 0; i < num_weight_blocks; i++) {
            int idx = indices[i];
            plan.blocks[idx].id = (uint32_t)idx;
            plan.blocks[idx].weight_index = (uint32_t)(idx * weights_per_block);
            plan.blocks[idx].num_weights = (uint32_t)weights_per_block;
            plan.blocks[idx].registry_format = (i < crit_count) ? crit_fmt : rest_fmt;
            RegFormat rf = plan.blocks[idx].registry_format;
            if (rf == RegFormat::OIL32)
                plan.blocks[idx].assigned_format = Format::OIL32;
            else if (rf == RegFormat::OIL16)
                plan.blocks[idx].assigned_format = Format::OIL16;
            else if (rf == RegFormat::OIL8)
                plan.blocks[idx].assigned_format = Format::OIL8;
            else if (rf == RegFormat::OIL4)
                plan.blocks[idx].assigned_format = Format::OIL4;
            else if (rf == RegFormat::OIL2)
                plan.blocks[idx].assigned_format = Format::OIL2;
            else if (rf == RegFormat::SPARK_Q0)
                plan.blocks[idx].assigned_format = Format::SPARK_Q0;
            else
                plan.blocks[idx].assigned_format = Format::OIL1;
            if (idx < (int)importance_scores_.size()) {
                plan.blocks[idx].importance_score = importance_scores_[idx];
            } else {
                plan.blocks[idx].importance_score = 0;
            }
        }
        // Count format distribution
        plan.num_oil32_blocks = plan.num_oil16_blocks = plan.num_oil8_blocks = 0;
        plan.num_oil4_blocks = plan.num_oil2_blocks = 0;
        plan.num_spark_blocks = plan.num_oil1_blocks = 0;
        for (const auto& b : plan.blocks) {
            if (b.assigned_format == Format::OIL32)      plan.num_oil32_blocks++;
            else if (b.assigned_format == Format::OIL16)  plan.num_oil16_blocks++;
            else if (b.assigned_format == Format::OIL8)   plan.num_oil8_blocks++;
            else if (b.assigned_format == Format::OIL4)   plan.num_oil4_blocks++;
            else if (b.assigned_format == Format::OIL2)   plan.num_oil2_blocks++;
            else if (b.assigned_format == Format::SPARK_Q0) plan.num_spark_blocks++;
            else plan.num_oil1_blocks++;
        }
    } else if (best_four_diff < single_diff && best_four_diff < best_two_diff) {
        plan.uses_mix = true;
        plan.selected_mix = best_four;

        int t1 = (int)std::round(best_four.tier1_ratio * num_weight_blocks);
        int t2 = (int)std::round(best_four.tier2_ratio * num_weight_blocks);
        int t3 = (int)std::round(best_four.tier3_ratio * num_weight_blocks);
        int t4 = num_weight_blocks - t1 - t2 - t3;
        if (t4 < 0) t4 = 0;

        std::vector<int> indices(num_weight_blocks);
        for (int i = 0; i < num_weight_blocks; i++) indices[i] = i;
        if (!importance_scores_.empty()) {
            std::sort(indices.begin(), indices.end(), [this](int a, int b) {
                float sa = a < (int)importance_scores_.size() ? importance_scores_[a] : 0;
                float sb = b < (int)importance_scores_.size() ? importance_scores_[b] : 0;
                return sa > sb;
            });
        }

        for (int i = 0; i < num_weight_blocks; i++) {
            int idx = indices[i];
            plan.blocks[idx].id = (uint32_t)idx;
            plan.blocks[idx].weight_index = (uint32_t)(idx * weights_per_block);
            plan.blocks[idx].num_weights = (uint32_t)weights_per_block;

            if (i < t1) {
                plan.blocks[idx].registry_format = best_four.tier1_fmt;
            } else if (i < t1 + t2) {
                plan.blocks[idx].registry_format = best_four.tier2_fmt;
            } else if (i < t1 + t2 + t3) {
                plan.blocks[idx].registry_format = best_four.tier3_fmt;
            } else {
                plan.blocks[idx].registry_format = best_four.tier4_fmt;
            }

            {
                RegFormat rf = plan.blocks[idx].registry_format;
                if (rf == RegFormat::OIL32)
                    plan.blocks[idx].assigned_format = Format::OIL32;
                else if (rf == RegFormat::OIL16)
                    plan.blocks[idx].assigned_format = Format::OIL16;
                else if (rf == RegFormat::OIL8)
                    plan.blocks[idx].assigned_format = Format::OIL8;
                else if (rf == RegFormat::OIL4)
                    plan.blocks[idx].assigned_format = Format::OIL4;
                else if (rf == RegFormat::OIL2)
                    plan.blocks[idx].assigned_format = Format::OIL2;
                else if (rf == RegFormat::SPARK_Q0)
                    plan.blocks[idx].assigned_format = Format::SPARK_Q0;
                else
                    plan.blocks[idx].assigned_format = Format::OIL1;
            }

            if (idx < (int)importance_scores_.size()) {
                plan.blocks[idx].importance_score = importance_scores_[idx];
            } else {
                plan.blocks[idx].importance_score = 0;
            }
        }
        // Count format distribution
        plan.num_oil32_blocks = plan.num_oil16_blocks = plan.num_oil8_blocks = 0;
        plan.num_oil4_blocks = plan.num_oil2_blocks = 0;
        plan.num_spark_blocks = plan.num_oil1_blocks = 0;
        for (const auto& b : plan.blocks) {
            if (b.assigned_format == Format::OIL32)      plan.num_oil32_blocks++;
            else if (b.assigned_format == Format::OIL16)  plan.num_oil16_blocks++;
            else if (b.assigned_format == Format::OIL8)   plan.num_oil8_blocks++;
            else if (b.assigned_format == Format::OIL4)   plan.num_oil4_blocks++;
            else if (b.assigned_format == Format::OIL2)   plan.num_oil2_blocks++;
            else if (b.assigned_format == Format::SPARK_Q0) plan.num_spark_blocks++;
            else plan.num_oil1_blocks++;
        }
    } else {
        std::vector<int> indices(num_weight_blocks);
        for (int i = 0; i < num_weight_blocks; i++) indices[i] = i;

        if (!importance_scores_.empty()) {
            std::sort(indices.begin(), indices.end(), [this](int a, int b) {
                float sa = a < (int)importance_scores_.size() ? importance_scores_[a] : 0;
                float sb = b < (int)importance_scores_.size() ? importance_scores_[b] : 0;
                return sa > sb;
            });
        }

        int o32 = 0, o16 = 0, o8 = 0, o4 = 0, o2 = 0, sp = 0, o1 = 0;
        compute_format_mix(num_weight_blocks, target_bpw_,
                           o32, o16, o8, o4, o2, sp, o1);

        plan.num_oil32_blocks = o32;
        plan.num_oil16_blocks = o16;
        plan.num_oil8_blocks = o8;
        plan.num_oil4_blocks = o4;
        plan.num_oil2_blocks = o2;
        plan.num_spark_blocks = sp;
        plan.num_oil1_blocks = o1;

        for (int i = 0; i < num_weight_blocks; i++) {
            int idx = indices[i];
            if (i < o32) {
                plan.blocks[idx].assigned_format = Format::OIL32;
                plan.blocks[idx].registry_format = RegFormat::OIL32;
            } else if (i < o32 + o16) {
                plan.blocks[idx].assigned_format = Format::OIL16;
                plan.blocks[idx].registry_format = RegFormat::OIL16;
            } else if (i < o32 + o16 + o8) {
                plan.blocks[idx].assigned_format = Format::OIL8;
                plan.blocks[idx].registry_format = RegFormat::OIL8;
            } else if (i < o32 + o16 + o8 + o4) {
                plan.blocks[idx].assigned_format = Format::OIL4;
                plan.blocks[idx].registry_format = RegFormat::OIL4;
            } else if (i < o32 + o16 + o8 + o4 + o2) {
                plan.blocks[idx].assigned_format = Format::OIL2;
                plan.blocks[idx].registry_format = RegFormat::OIL2;
            } else if (i < o32 + o16 + o8 + o4 + o2 + sp) {
                plan.blocks[idx].assigned_format = Format::SPARK_Q0;
                plan.blocks[idx].registry_format = RegFormat::SPARK_Q0;
            } else {
                plan.blocks[idx].assigned_format = Format::OIL1;
                plan.blocks[idx].registry_format = RegFormat::OIL1;
            }
            if (idx < (int)importance_scores_.size()) {
                plan.blocks[idx].importance_score = importance_scores_[idx];
            } else {
                plan.blocks[idx].importance_score = 0;
            }
        }
    }

    plan.achieved_bpw = estimate_bpw(plan);
    return plan;
}

FormatPlan FormatPlanner::plan_for_model(int64_t num_weights) {
    return allocate((int)(num_weights / 256), 256);
}

const std::vector<float>& FormatPlanner::importance_scores() const {
    return importance_scores_;
}

float FormatPlanner::estimate_bpw(const FormatPlan& plan) {
    if (plan.blocks.empty()) return 0;
    float total = 0;
    for (const auto& b : plan.blocks) {
        total += format_bpw(b.assigned_format);
    }
    return total / (float)plan.blocks.size();
}

FormatPlan FormatPlanner::plan_for_target(float target_bpw, int num_blocks,
                                           int weights_per_block) {
    FormatPlanner planner(target_bpw);
    return planner.allocate(num_blocks, weights_per_block);
}

} // namespace oil
