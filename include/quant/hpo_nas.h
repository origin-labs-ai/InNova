#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/optimizer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstdint>

namespace quant {
namespace hpo_nas {

struct SearchSpace {
    float lr_min = 1e-5f;
    float lr_max = 1e-1f;
    float wd_min = 1e-6f;
    float wd_max = 1e-1f;
    float beta1_min = 0.7f;
    float beta1_max = 0.999f;
    int64_t warmup_min = 0;
    int64_t warmup_max = 1000;
    int64_t batch_size_min = 1;
    int64_t batch_size_max = 256;
    int depth_min = 1;
    int depth_max = 48;
    int width_min = 64;
    int width_max = 4096;
    int heads_min = 1;
    int heads_max = 64;
    float ffn_ratio_min = 1.0f;
    float ffn_ratio_max = 8.0f;
    float dropout_min = 0.0f;
    float dropout_max = 0.5f;
};

struct HPConfig {
    int64_t id = 0;
    float lr = 3e-4f;
    float weight_decay = 1e-2f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    int64_t warmup_steps = 100;
    int64_t batch_size = 32;
    float dropout = 0.1f;
    int depth = 12;
    int width = 768;
    int num_heads = 12;
    float ffn_ratio = 4.0f;
    float score = 0.0f;
    float val_loss = 1e10f;
    float train_loss = 1e10f;
    int epoch = 0;
    bool alive = true;
};

struct Architecture {
    int64_t id = 0;
    int num_layers = 12;
    int hidden_size = 768;
    int num_heads = 12;
    int ffn_size = 3072;
    int vocab_size = 32000;
    float dropout = 0.1f;
    float score = 0.0f;
    float val_loss = 1e10f;
    float latency_ms = 0.0f;
    int64_t param_count = 0;
    int64_t memory_bytes = 0;
    bool alive = true;
};

struct NASMutateOp {
    enum Type : uint8_t {
        ADD_LAYER = 0,
        REMOVE_LAYER = 1,
        CHANGE_WIDTH = 2,
        CHANGE_HEADS = 3,
        CHANGE_FFN = 4,
        CHANGE_DROPOUT = 5,
        CHANGE_VOCAB = 6
    };
    Type type = ADD_LAYER;
    float magnitude = 0.1f;
};

struct HardwareConstraints {
    float target_latency_ms = 10.0f;
    int64_t target_memory_bytes = 1024 * 1024 * 1024;
    int64_t max_params = 1000000000;
    bool enabled = false;
};

struct BenchmarkResult {
    float accuracy = 0.0f;
    float loss = 1e10f;
    double time_ms = 0.0;
    int64_t param_count = 0;
    int64_t memory_bytes = 0;
};

struct SuccessiveHalvingConfig {
    int n_configs = 0;
    int n_rungs = 0;
    float reduction_factor = 3.0f;
    std::vector<int> configs_per_rung;
    std::vector<double> budget_per_rung;
};

class HPOEngine {
public:
    HPOEngine(Trainer* trainer = nullptr, Model* model = nullptr);
    ~HPOEngine();

    void set_search_space(const SearchSpace& space);
    SearchSpace get_search_space() const;
    void set_hardware_constraints(const HardwareConstraints& hc);

    HPConfig random_sample();
    std::vector<HPConfig> random_search(int n_configs);

    std::vector<HPConfig> bayesian_optimization(int n_initial = 5, int n_iterations = 20);
    float surrogate_predict(const HPConfig& config);
    void surrogate_update(const HPConfig& config, float score);
    float acquisition_function(const HPConfig& config, int n_evaluations) const;

    void population_based_training(int n_population = 8, int n_generations = 10);
    HPConfig select_parent(const std::vector<HPConfig>& population);
    void exploit_and_explore(HPConfig& child, const HPConfig& parent, float perturbation_factor);

    SuccessiveHalvingConfig setup_successive_halving(int n_configs = 27, int n_rungs = 5);
    std::vector<HPConfig> successive_halving(int n_configs = 27, int n_rungs = 5);

    HPConfig optimize(int n_budget = 100, const std::string& method = "pbt");

    void record_eval(const HPConfig& config, float score);
    std::vector<std::pair<HPConfig, float>> get_eval_history() const;

    bool save_best_configs(const std::string& path, int top_k = 5) const;
    std::vector<HPConfig> load_configs(const std::string& path) const;

    HPConfig get_best() const;
    std::vector<HPConfig> get_top_k(int k) const;

    void set_trainer(Trainer* trainer);
    void set_model(Model* model);

private:
    Trainer* trainer_;
    Model* model_;
    SearchSpace space_;
    HardwareConstraints hw_constraints_;
    std::vector<std::pair<HPConfig, float>> eval_history_;
    std::vector<std::pair<std::vector<float>, float>> gp_training_data_;
    int64_t next_config_id_ = 0;

    float evaluate_config(const HPConfig& config);
    float compute_hardware_penalty(const Architecture& arch) const;
    HPConfig crossover(const HPConfig& a, const HPConfig& b);
    HPConfig mutate_config(const HPConfig& config, float mutation_rate);
    float compute_distance(const HPConfig& a, const HPConfig& b) const;
    float rbf_kernel(const float* x1, const float* x2, int n, float length_scale = 1.0f) const;
    std::vector<float> config_to_features(const HPConfig& config) const;
    int64_t estimate_param_count(int depth, int width, int ffn_size, int vocab_size) const;
    int64_t estimate_memory_bytes(int64_t params) const;
    double estimate_latency_ms(int depth, int width, int num_heads) const;
};

// ========================================================================
// NAS Engine
// ========================================================================

struct NASConfig {
    int population = 50;
    int generations = 20;
    int tournament_size = 3;
    float mutation_rate = 0.3f;
    float crossover_rate = 0.7f;
    bool hardware_aware = false;
    bool quant_native = true;
};

class NASEngine {
public:
    NASEngine(Model* model = nullptr);
    ~NASEngine();

    void set_search_config(const NASConfig& config);
    void set_hardware_constraints(const HardwareConstraints& hc);

    Architecture random_architecture();
    std::vector<Architecture> initialize_population(int n);

    Architecture mutate(const Architecture& arch);
    Architecture crossover(const Architecture& a, const Architecture& b);
    float evaluate(const Architecture& arch);

    Architecture tournament_select(const std::vector<Architecture>& pop);
    Architecture search(int population = 50, int generations = 20);

    Architecture darts_search(int n_iterations = 50);
    float darts_evaluate(const Architecture& arch);
    Architecture darts_migrate(const Architecture& arch, float alpha = 0.1f);

    float hardware_aware_score(const Architecture& arch);
    int64_t estimate_latency_cycles(int depth, int width, int num_heads) const;
    int64_t estimate_memory(int depth, int width, int ffn_size) const;

    Architecture mutate_add_layer(const Architecture& arch);
    Architecture mutate_remove_layer(const Architecture& arch);
    Architecture mutate_change_width(const Architecture& arch);
    Architecture mutate_change_heads(const Architecture& arch);
    Architecture mutate_change_ffn(const Architecture& arch);
    Architecture mutate_change_dropout(const Architecture& arch);

    void record_architecture(const Architecture& arch);
    Architecture get_best_architecture() const;
    std::vector<Architecture> get_pareto_front() const;

    bool save_architectures(const std::string& path, int top_k = 10) const;
    std::vector<Architecture> load_architectures(const std::string& path) const;

    void set_model(Model* model);

private:
    Model* model_;
    NASConfig config_;
    HardwareConstraints hw_constraints_;
    std::vector<Architecture> history_;
    int64_t next_arch_id_ = 0;

    float compute_fitness(const Architecture& arch) const;
    bool is_valid(const Architecture& arch) const;
    NASMutateOp random_mutation_op() const;
    float quant_format_penalty(const Architecture& arch) const;
    int64_t compute_param_count(const Architecture& arch) const;
};

} // namespace hpo_nas
} // namespace quant
