#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <random>
#include "oil/types.h"
#include "oil/model.h"
#include "oil/format_registry.h"

namespace oil {
namespace adapters {

enum class TrainingMode : uint8_t {
    SCRATCH,
    CONTINUAL
};

struct ContinualConfig {
    TrainingMode mode = TrainingMode::SCRATCH;

    std::size_t replay_capacity = 65536;
    float fisher_bias = 0.5f;
    std::string spill_path = "./replay_spill";

    float ecc_lambda = 1.0f;
    float ecc_freeze_threshold = 0.1f;

    float distill_alpha = 0.5f;
    float distill_temperature = 2.0f;

    float stability_lr_factor = 0.5f;
    float plasticity_lr_factor = 2.0f;
    int forgetting_check_interval = 100;

    size_t batch_size = 8;
    size_t seq_length = 512;
    size_t max_steps = 10000;
    float learning_rate = 1e-4f;
    int log_interval = 10;
};

struct ContinualMetrics {
    double total_loss = 0.0;
    double new_data_loss = 0.0;
    double replay_loss = 0.0;
    double distill_loss = 0.0;
    double ecc_regularizer = 0.0;
    float forgetting_weight = 0.0f;
    float backward_transfer = 0.0f;
    std::size_t replay_buffer_size = 0;
    std::size_t frozen_count = 0;
    std::size_t step = 0;
};

struct ReplaySample {
    std::vector<float> input;
    std::vector<float> target;
    float fisher_weight = 1.0f;
    std::uint32_t task_id = 0;
};

class ContinualTrainer {
public:
    explicit ContinualTrainer(DenseModel* model, const ContinualConfig& cfg = {});
    ~ContinualTrainer() = default;

    void train(const std::vector<std::vector<float>>& train_data,
               const std::vector<std::vector<float>>& eval_data = {});

    ContinualMetrics train_step(const float* input, const float* target,
                                std::size_t batch_size, std::size_t seq_len);

    void on_task_boundary(std::size_t new_task_id);

    float compute_forgetting_weight() const;
    float compute_backward_transfer() const;

    [[nodiscard]] TrainingMode mode() const { return cfg_.mode; }
    [[nodiscard]] const ContinualMetrics& last_metrics() const { return last_; }
    [[nodiscard]] std::size_t current_step() const { return step_; }

private:
    ContinualMetrics train_step_continual(const float* input, const float* target,
                                          std::size_t batch_size, std::size_t seq_len);
    void update_stability_plasticity(float fw_estimate);
    void consolidate_anchors();
    void accumulate_fisher(const float* grads, std::size_t n);
    float ecc_regularizer() const;
    void distillation_loss(const float* current_logits, const float* old_logits,
                           std::size_t n, float& loss_out) const;
    void replay_insert(ReplaySample s);
    void importance_sample(std::vector<std::size_t>& indices, std::size_t n) const;

    DenseModel* model_;
    ContinualConfig cfg_;
    ContinualMetrics last_{};

    std::vector<float> fisher_;
    std::vector<float> anchor_;
    bool anchored_ = false;

    std::vector<float> old_logits_;
    bool has_old_logits_ = false;

    std::vector<ReplaySample> replay_;
    std::size_t replay_seen_ = 0;

    std::vector<float> task_perf_;
    std::size_t current_task_ = 0;
    float current_lr_factor_ = 1.0f;
    std::size_t step_ = 0;
    std::mt19937_64 rng_{42};
};

} // namespace adapters
} // namespace oil
