#pragma once
#include "quant/tensor.h"
#include "quant/optimizer.h"
#include <vector>
#include <string>
#include <functional>

namespace quant {

struct Comparison {
    std::vector<int> chosen_ids;
    std::vector<int> rejected_ids;
    std::string prompt;
    float reward_chosen = 0.0f;
    float reward_rejected = 0.0f;
};

class RewardModel {
public:
    RewardModel() = default;
    RewardModel(int64_t hidden_size, int64_t num_layers = 2, float importance_factor = 8.0f);
    RewardModel(const RewardModel&) = delete;
    RewardModel& operator=(const RewardModel&) = delete;
    RewardModel(RewardModel&&) = default;
    RewardModel& operator=(RewardModel&&) = default;

    Tensor score_pair(const Tensor& chosen, const Tensor& rejected);
    Tensor score(const Tensor& sequence);
    Tensor reward_loss(const Tensor& chosen_rewards, const Tensor& rejected_rewards);
    float train_step(const Tensor& chosen, const Tensor& rejected, Optimizer* opt, float loss_scale = 1.0f);
    float compute_abstention_reward(const std::string& output, float confidence);

    std::vector<Tensor*> parameters();
    void load(const std::string& path);
    void save(const std::string& path) const;

    void set_log_callback(std::function<void(float loss, float acc)> cb) { log_cb_ = cb; }
    int64_t hidden_size() const { return hidden_size_; }
    int64_t num_layers() const { return num_layers_; }
    float last_loss() const { return last_loss_; }

private:
    int64_t hidden_size_ = 0;
    int64_t num_layers_ = 2;
    float importance_factor_ = 8.0f;

    Tensor fc1_weight_, fc1_bias_;
    Tensor fc2_weight_, fc2_bias_;
    Tensor fc3_weight_, fc3_bias_;
    float last_loss_ = 0.0f;

    Tensor forward_mlp(const Tensor& x);
    Tensor sigmoid_op(const Tensor& x);

    std::function<void(float loss, float acc)> log_cb_;
};

struct RLHFMetrics {
    float reward_loss = 0.0f;
    float reward_accuracy = 0.0f;
    float ppo_policy_loss = 0.0f;
    float ppo_value_loss = 0.0f;
    float ppo_entropy = 0.0f;
    float ppo_kl = 0.0f;
    float ppo_total_reward = 0.0f;
    float dpo_loss = 0.0f;
    float dpo_kl = 0.0f;
    float kl_divergence = 0.0f;
    float mean_reward = 0.0f;
    float min_reward = 0.0f;
    float max_reward = 0.0f;
    int step = 0;
    int n_comparisons = 0;
};

} // namespace quant
