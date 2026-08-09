#pragma once
#include "quant/tensor.h"
#include "quant/types.h"
#include <vector>
#include <string>

namespace quant {

// F1: Expert capacity tuning
struct ExpertCapacity {
    std::vector<int64_t> capacities;
    static ExpertCapacity adaptive(int64_t num_experts, int64_t tokens,
                                    const float* importance);
};

// F2: DS-MoE prune — convert dense layers to MoE via pruning
class DenseToMoEPruner {
public:
    DenseToMoEPruner(int64_t hidden, int64_t n_experts, float sparsity = 0.5f);
    Tensor prune_ffn(const Tensor& gate, const Tensor& up, const Tensor& down);
private:
    int64_t hidden_, n_experts_;
    float sparsity_;
};

// F3: MoE + QUANT format — quantized experts
class MoEQUANTFormat {
public:
    static void save_experts(const std::vector<Tensor>& experts,
                              const std::string& path);
    static std::vector<Tensor> load_experts(const std::string& path);
};

// F4: Expert dropout
class ExpertDropout {
public:
    ExpertDropout(float rate = 0.1f);
    int64_t get_active_count(int64_t total) const;
    std::vector<bool> get_mask(int64_t total) const;
private:
    float rate_;
};

// F5: Balance monitoring
struct ExpertBalance {
    std::vector<int64_t> token_counts;
    float load_balance_loss;
    float stddev;
};
ExpertBalance compute_balance(const std::vector<int64_t>& assignments, int64_t n_experts);

// F6: Switch v2 — improved routing with expert temperature
class SwitchV2Router {
public:
    SwitchV2Router(int64_t n_experts, float capacity_factor = 1.25f);
    Tensor forward(const Tensor& logits, int64_t tokens);
private:
    int64_t n_experts_;
    float capacity_factor_;
};

// F7: Expert choice v2 — tokens choose experts, experts also choose tokens
class ExpertChoiceV2 {
public:
    ExpertChoiceV2(int64_t n_experts, int64_t top_k = 2);
    Tensor forward(const Tensor& x, const Tensor& logits);
private:
    int64_t n_experts_, top_k_;
};

// F8: Expert merging — combine similar experts via SVD
class ExpertMerger {
public:
    ExpertMerger(float similarity_threshold = 0.9f);
    std::vector<Tensor> merge(const std::vector<Tensor>& experts);
private:
    float threshold_;
    float cosine_sim(const float* a, const float* b, int64_t n);
};

// F9: Dynamic expert count — grow/shrink experts at runtime
class DynamicExpertPool {
public:
    DynamicExpertPool(int64_t initial, int64_t max_experts);
    int64_t add_expert(const Tensor& weights);
    bool remove_expert(int64_t idx);
    int64_t count() const { return (int64_t)experts_.size(); }
private:
    std::vector<Tensor> experts_;
    int64_t max_experts_;
};

// F10: Expert parallelism — distribute experts across devices
class ExpertParallel {
public:
    ExpertParallel(int64_t world_size, int64_t world_rank);
    Tensor forward(const Tensor& x, int64_t n_experts);
private:
    int64_t world_size_, world_rank_;
};

} // namespace quant
