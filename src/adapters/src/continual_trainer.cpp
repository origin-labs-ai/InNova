#include "adapters/continual_trainer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace oil {
namespace adapters {

ContinualTrainer::ContinualTrainer(DenseModel* model, const ContinualConfig& cfg)
    : model_(model), cfg_(cfg) {
    if (cfg_.mode == TrainingMode::CONTINUAL) {
        replay_.reserve(cfg_.replay_capacity);
    }
}

void ContinualTrainer::train(const std::vector<std::vector<float>>& train_data,
                              const std::vector<std::vector<float>>& eval_data) {
    (void)eval_data;
    if (cfg_.mode == TrainingMode::SCRATCH) {
        for (std::size_t s = 0; s < cfg_.max_steps; ++s) {
            std::size_t idx = s % train_data.size();
            if (train_data[idx].size() < 2) continue;
            std::size_t half = train_data[idx].size() / 2;
            train_step(train_data[idx].data(), train_data[idx].data() + half,
                       cfg_.batch_size, cfg_.seq_length);
        }
    } else {
        for (std::size_t s = 0; s < cfg_.max_steps; ++s) {
            std::size_t idx = s % train_data.size();
            if (train_data[idx].size() < 2) continue;
            std::size_t half = train_data[idx].size() / 2;
            auto m = train_step(train_data[idx].data(), train_data[idx].data() + half,
                                cfg_.batch_size, cfg_.seq_length);
            if (cfg_.forgetting_check_interval > 0 &&
                step_ % static_cast<std::size_t>(cfg_.forgetting_check_interval) == 0) {
                float fw = compute_forgetting_weight();
                update_stability_plasticity(fw);
            }
            ++step_;
        }
    }
}

ContinualMetrics ContinualTrainer::train_step(const float* input, const float* target,
                                               std::size_t batch_size, std::size_t seq_len) {
    if (cfg_.mode == TrainingMode::SCRATCH) {
        last_.step = step_++;
        last_.total_loss = 0.0;
        return last_;
    }
    return train_step_continual(input, target, batch_size, seq_len);
}

ContinualMetrics ContinualTrainer::train_step_continual(
        const float* input, const float* target,
        std::size_t batch_size, std::size_t seq_len) {
    std::size_t flat = batch_size * seq_len;

    double new_loss = 0.0;
    double replay_l = 0.0;
    double distill_l = 0.0;
    double ecc_reg = 0.0;

    for (std::size_t i = 0; i < flat; ++i) {
        new_loss += (double)input[i] * (double)input[i];
    }
    new_loss /= static_cast<double>(flat);

    if (!replay_.empty()) {
        std::size_t rbatch = (std::min)(flat, replay_.size());
        std::vector<std::size_t> indices;
        importance_sample(indices, rbatch);
        for (std::size_t idx : indices) {
            const auto& s = replay_[idx];
            std::size_t sz = (std::min)(s.input.size(), flat);
            for (std::size_t j = 0; j < sz; ++j) {
                replay_l += (double)(input[j] - s.input[j]) * (double)(input[j] - s.input[j]);
            }
        }
        replay_l /= static_cast<double>(rbatch);
    }

    if (has_old_logits_ && !old_logits_.empty()) {
        std::size_t n = (std::min)(old_logits_.size(), flat);
        float distill_f = 0.0f;
        distillation_loss(input, old_logits_.data(), n, distill_f);
        distill_l = static_cast<double>(distill_f);
    }

    ecc_reg = ecc_regularizer();

    double total = new_loss +
                   (double)cfg_.ecc_lambda * ecc_reg +
                   (replay_.empty() ? 0.0 : 0.1 * replay_l) +
                   (has_old_logits_ ? (double)(1.0f - cfg_.distill_alpha) * distill_l : 0.0);

    last_.total_loss = total;
    last_.new_data_loss = new_loss;
    last_.replay_loss = replay_l;
    last_.distill_loss = distill_l;
    last_.ecc_regularizer = ecc_reg;
    last_.replay_buffer_size = replay_.size();
    last_.frozen_count = 0;
    last_.step = step_;

    ReplaySample sample;
    sample.input.assign(input, input + flat);
    sample.target.assign(target, target + flat);
    sample.fisher_weight = 1.0f;
    sample.task_id = static_cast<uint32_t>(current_task_);
    replay_insert(std::move(sample));

    if (!has_old_logits_) {
        old_logits_.assign(input, input + flat);
        has_old_logits_ = true;
    }

    ++step_;
    return last_;
}

void ContinualTrainer::on_task_boundary(std::size_t new_task_id) {
    if (cfg_.mode == TrainingMode::SCRATCH) return;

    task_perf_.push_back(1.0f);
    consolidate_anchors();
    old_logits_.clear();
    has_old_logits_ = false;
    current_task_ = new_task_id;
}

float ContinualTrainer::compute_forgetting_weight() const {
    if (task_perf_.size() < 2) return 0.0f;
    float max_drop = 0.0f;
    for (std::size_t k = 0; k + 1 < task_perf_.size(); ++k) {
        float drop = task_perf_[k] - task_perf_.back();
        if (drop > max_drop) max_drop = drop;
    }
    return max_drop;
}

float ContinualTrainer::compute_backward_transfer() const {
    if (task_perf_.size() < 2) return 0.0f;
    float bwt = 0.0f;
    for (std::size_t k = 0; k + 1 < task_perf_.size(); ++k) {
        bwt += task_perf_.back() - task_perf_[k];
    }
    return bwt / static_cast<float>(task_perf_.size() - 1);
}

void ContinualTrainer::update_stability_plasticity(float fw_estimate) {
    if (fw_estimate > 0.1f) {
        current_lr_factor_ *= cfg_.stability_lr_factor;
        if (current_lr_factor_ < 0.01f) current_lr_factor_ = 0.01f;
    } else {
        current_lr_factor_ *= cfg_.plasticity_lr_factor;
        if (current_lr_factor_ > 10.0f) current_lr_factor_ = 10.0f;
    }
}

void ContinualTrainer::consolidate_anchors() {
    if (!model_) return;
    std::vector<Tensor*> params;
    model_->get_parameters(params);
    std::size_t total = 0;
    for (const auto* p : params) {
        total += static_cast<std::size_t>(p->numel());
    }
    if (total == 0) return;

    anchor_.clear();
    anchor_.reserve(total);
    for (const auto* p : params) {
        const float* d = static_cast<const float*>(p->data());
        for (int64_t i = 0; i < p->numel(); ++i) {
            anchor_.push_back(d[i]);
        }
    }

    if (fisher_.size() != anchor_.size()) {
        fisher_.assign(anchor_.size(), 0.0f);
    }
    anchored_ = true;
}

void ContinualTrainer::accumulate_fisher(const float* grads, std::size_t n) {
    if (fisher_.size() != n) fisher_.assign(n, 0.0f);
    const float mom = 0.9f;
    for (std::size_t i = 0; i < n; ++i) {
        fisher_[i] = mom * fisher_[i] + (1.0f - mom) * grads[i] * grads[i];
    }
}

float ContinualTrainer::ecc_regularizer() const {
    if (!anchored_ || !model_) return 0.0f;
    std::vector<Tensor*> params;
    model_->get_parameters(params);
    std::size_t offset = 0;
    double acc = 0.0;
    for (const auto* p : params) {
        const float* cur = static_cast<const float*>(p->data());
        std::size_t n = static_cast<std::size_t>(p->numel());
        for (std::size_t i = 0; i < n && (offset + i) < anchor_.size(); ++i) {
            double d = static_cast<double>(cur[i]) - static_cast<double>(anchor_[offset + i]);
            double f = (offset + i) < fisher_.size() ? fisher_[offset + i] : 0.0;
            acc += f * d * d;
        }
        offset += n;
    }
    return static_cast<float>(acc);
}

void ContinualTrainer::distillation_loss(const float* current_logits,
                                          const float* old_logits,
                                          std::size_t n, float& loss_out) const {
    double sum = 0.0;
    double temp_sq = static_cast<double>(cfg_.distill_temperature) *
                     static_cast<double>(cfg_.distill_temperature);
    for (std::size_t i = 0; i < n; ++i) {
        double c = static_cast<double>(current_logits[i]) / temp_sq;
        double o = static_cast<double>(old_logits[i]) / temp_sq;
        double diff = c - o;
        sum += diff * diff;
    }
    loss_out = static_cast<float>(sum / static_cast<double>(n));
}

void ContinualTrainer::replay_insert(ReplaySample s) {
    ++replay_seen_;
    if (replay_.size() < cfg_.replay_capacity) {
        replay_.push_back(std::move(s));
        return;
    }
    std::uniform_int_distribution<std::size_t> dist(0, replay_seen_ - 1);
    std::size_t r = dist(rng_);
    if (r < replay_.size()) {
        replay_[r] = std::move(s);
    }
}

void ContinualTrainer::importance_sample(std::vector<std::size_t>& indices,
                                          std::size_t n) const {
    indices.clear();
    if (replay_.empty()) return;
    indices.reserve(n);

    std::vector<double> cdf(replay_.size(), 0.0);
    double acc = 0.0;
    for (std::size_t i = 0; i < replay_.size(); ++i) {
        acc += std::pow(std::max(replay_[i].fisher_weight, 1e-8f),
                        static_cast<double>(cfg_.fisher_bias));
        cdf[i] = acc;
    }

    std::mt19937_64 engine = rng_;
    std::uniform_real_distribution<double> u(0.0, acc);
    for (std::size_t k = 0; k < n; ++k) {
        double r = u(engine);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        std::size_t idx = (it == cdf.end()) ? replay_.size() - 1
                                             : static_cast<std::size_t>(it - cdf.begin());
        indices.push_back(idx);
    }
}

} // namespace adapters
} // namespace oil
