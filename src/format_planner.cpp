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
                                        int& oil8, int& oil4,
                                        int& spark, int& oil1) {
    const float bpw_spark = 1.50f;
    const float bpw_oil4 = 4.0f;
    const float bpw_oil8 = 8.0f;
    const float regime_switch = 2.25f;

    float f_oil8 = 0.0f, f_oil4 = 0.0f, f_spark = 0.0f;

    if (target_bpw >= bpw_oil8) {
        f_oil8 = 1.0f;
    } else if (target_bpw >= bpw_oil4) {
        f_oil8 = (target_bpw - bpw_oil4) / (bpw_oil8 - bpw_oil4);
        f_oil4 = 1.0f - f_oil8;
    } else if (target_bpw >= regime_switch) {
        f_oil4 = (target_bpw - bpw_spark) / (bpw_oil4 - bpw_spark);
        f_spark = 1.0f - f_oil4;
    } else {
        f_oil8 = (target_bpw - bpw_spark) / (bpw_oil8 - bpw_spark);
        if (f_oil8 < 0.0f) f_oil8 = 0.0f;
        f_spark = 1.0f - f_oil8;
    }

    oil8 = (int)std::round(f_oil8 * (double)num_blocks);
    oil4 = (int)std::round(f_oil4 * (double)num_blocks);
    spark = (int)std::round(f_spark * (double)num_blocks);
    oil1 = num_blocks - oil8 - oil4 - spark;
    if (oil1 < 0) oil1 = 0;

    oil8 = (std::min)(oil8, num_blocks);
    oil4 = (std::min)(oil4, num_blocks - oil8);
    spark = (std::min)(spark, num_blocks - oil8 - oil4);
    oil1 = num_blocks - oil8 - oil4 - spark;
}

static RegFormat bpw_to_reg_format(float bpw) {
    if (bpw >= 31.99f) return RegFormat::OIL32;
    if (bpw >= 15.99f) return RegFormat::OIL16;
    if (bpw >= 7.99f)  return RegFormat::OIL8;
    if (bpw >= 3.99f)  return RegFormat::OIL4;
    if (bpw >= 1.57f)  return RegFormat::SPARK_Q0;
    return RegFormat::OIL1;
}

FormatPlan FormatPlanner::allocate(int num_weight_blocks, int weights_per_block) {
    FormatPlan plan;
    plan.target_bpw = target_bpw_;
    plan.blocks.resize(num_weight_blocks);
    plan.num_oil8_blocks = 0;
    plan.num_oil4_blocks = 0;
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
            if (rf == RegFormat::OIL8 || rf == RegFormat::OIL16 || rf == RegFormat::OIL32)
                plan.blocks[idx].assigned_format = Format::OIL8;
            else if (rf == RegFormat::OIL4)
                plan.blocks[idx].assigned_format = Format::OIL4;
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
                if (rf == RegFormat::OIL32 || rf == RegFormat::OIL16 || rf == RegFormat::OIL8)
                    plan.blocks[idx].assigned_format = Format::OIL8;
                else if (rf == RegFormat::OIL4 || rf == RegFormat::OIL2)
                    plan.blocks[idx].assigned_format = Format::OIL4;
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

        int oil8_count = 0, oil4_count = 0, spark_count = 0, oil1_count = 0;
        compute_format_mix(num_weight_blocks, target_bpw_,
                           oil8_count, oil4_count, spark_count, oil1_count);

        plan.num_oil8_blocks = oil8_count;
        plan.num_oil4_blocks = oil4_count;
        plan.num_spark_blocks = spark_count;
        plan.num_oil1_blocks = oil1_count;

        for (int i = 0; i < num_weight_blocks; i++) {
            int idx = indices[i];
            plan.blocks[idx].id = (uint32_t)idx;
            plan.blocks[idx].weight_index = (uint32_t)(idx * weights_per_block);
            plan.blocks[idx].num_weights = (uint32_t)weights_per_block;
            if (i < oil8_count) {
                plan.blocks[idx].assigned_format = Format::OIL8;
                plan.blocks[idx].registry_format = RegFormat::OIL8;
            } else if (i < oil8_count + oil4_count) {
                plan.blocks[idx].assigned_format = Format::OIL4;
                plan.blocks[idx].registry_format = RegFormat::OIL4;
            } else if (i < oil8_count + oil4_count + spark_count) {
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
