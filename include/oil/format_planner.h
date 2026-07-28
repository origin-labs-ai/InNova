#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include "oil/format_registry.h"
#include <vector>
#include <string>

namespace oil {

struct WeightBlock {
    uint32_t id;
    uint32_t weight_index;
    uint32_t num_weights;
    Format assigned_format;
    RegFormat registry_format;
    float importance_score;
};

struct FormatPlan {
    float target_bpw;
    float achieved_bpw;
    std::vector<WeightBlock> blocks;

    int num_oil32_blocks;
    int num_oil16_blocks;
    int num_oil8_blocks;
    int num_oil4_blocks;
    int num_oil2_blocks;
    int num_spark_blocks;
    int num_oil1_blocks;

    FormatDescriptor selected_single;
    MixDescriptor selected_mix;
    bool uses_mix;
};

enum class ImportanceMetric {
    MAGNITUDE,
    FISHER_DIAG,
    // HESSIAN_DIAG: second-order diagonal — reserved, not yet implemented
};

class FormatPlanner {
public:
    FormatPlanner(float target_bpw = 1.50f);
    
    void score_importance(const Tensor& weights,
                          const Tensor& calibration_activations,
                          int block_size = 256,
                          ImportanceMetric metric = ImportanceMetric::MAGNITUDE);
    
    void score_importance_fisher(const Tensor& weights,
                                 const Tensor& gradients,
                                 int block_size = 256);
    
    FormatPlan allocate(int num_weight_blocks, int weights_per_block = 256);
    
    FormatPlan plan_for_model(int64_t num_weights);
    
    const std::vector<float>& importance_scores() const;
    
    void set_target_bpw(float bpw) { target_bpw_ = bpw; }
    float target_bpw() const { return target_bpw_; }
    
    static float estimate_bpw(const FormatPlan& plan);
    
    static void compute_format_mix(int num_blocks, float target_bpw,
                                   int& oil32, int& oil16, int& oil8,
                                   int& oil4, int& oil2, int& spark, int& oil1);

    static FormatPlan plan_for_target(float target_bpw, int num_blocks,
                                      int weights_per_block = 256);
    
private:
    float target_bpw_;
    std::vector<float> importance_scores_;
};

} // namespace oil
