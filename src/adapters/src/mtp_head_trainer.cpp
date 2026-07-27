#include "adapters/mtp_head_trainer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace oil {
namespace adapters {

MTPHeadTrainer::MTPHeadTrainer(const MTPHeadConfig& cfg)
    : cfg_(cfg) {}

void MTPHeadTrainer::initialize() {
    heads_.clear();
    momentum_.clear();
    velocity_.clear();
    heads_.resize(cfg_.num_heads);
    momentum_.resize(cfg_.num_heads);
    velocity_.resize(cfg_.num_heads);

    for (std::size_t h = 0; h < cfg_.num_heads; ++h) {
        auto& head = heads_[h];
        head.projection_rows = cfg_.hidden_size;
        head.projection_cols = cfg_.hidden_size;
        std::size_t n = head.projection_rows * head.projection_cols;

        head.projection.resize(n);
        std::uniform_real_distribution<float> dist(
            -1.0f / std::sqrt(static_cast<float>(cfg_.hidden_size)),
             1.0f / std::sqrt(static_cast<float>(cfg_.hidden_size)));
        for (std::size_t i = 0; i < n; ++i) {
            head.projection[i] = dist(rng_);
        }
        head.output_bias.resize(cfg_.vocab_size, 0.0f);
        head.scale = 1.0f;
        head.is_quantized = false;

        momentum_[h].assign(n, 0.0f);
        velocity_[h].assign(n, 0.0f);
    }
    metrics_.current_draft_k = 3;
}

std::vector<float> MTPHeadTrainer::train_step(const float* hidden_states,
                                                const float* targets,
                                                std::size_t batch_size,
                                                std::size_t seq_len,
                                                std::size_t hidden_size) {
    std::size_t batch_seq = batch_size * seq_len;
    std::vector<float> input_grad(batch_seq * hidden_size, 0.0f);
    float total_loss = 0.0f;

    for (std::size_t h = 0; h < heads_.size(); ++h) {
        std::vector<float> logits(batch_seq * cfg_.vocab_size);
        head_forward(hidden_states, batch_seq, hidden_size, h, logits.data());

        float loss = 0.0f;
        std::vector<float> grad_logits(batch_seq * cfg_.vocab_size, 0.0f);
        for (std::size_t i = 0; i < batch_seq; ++i) {
            std::size_t t_idx = static_cast<std::size_t>(targets[i]);
            if (t_idx >= cfg_.vocab_size) t_idx = 0;
            float max_logit = -1e30f;
            for (std::size_t v = 0; v < cfg_.vocab_size; ++v) {
                float val = logits[i * cfg_.vocab_size + v];
                if (val > max_logit) max_logit = val;
            }
            float sum_exp = 0.0f;
            for (std::size_t v = 0; v < cfg_.vocab_size; ++v) {
                float exp_val = std::exp(logits[i * cfg_.vocab_size + v] - max_logit);
                grad_logits[i * cfg_.vocab_size + v] = exp_val;
                sum_exp += exp_val;
            }
            for (std::size_t v = 0; v < cfg_.vocab_size; ++v) {
                grad_logits[i * cfg_.vocab_size + v] /= sum_exp;
            }
            loss += -logits[i * cfg_.vocab_size + t_idx] + max_logit + std::log(sum_exp);
            grad_logits[i * cfg_.vocab_size + t_idx] -= 1.0f;
        }
        loss /= static_cast<float>(batch_seq);
        total_loss += loss;
        for (auto& g : grad_logits) g /= static_cast<float>(batch_seq);

        std::vector<float> hgrad(batch_seq * hidden_size, 0.0f);
        head_backward(hidden_states, grad_logits.data(), batch_seq, hidden_size, h, hgrad.data());
        for (std::size_t i = 0; i < hgrad.size(); ++i) {
            input_grad[i] += hgrad[i];
        }
        update_weights(h, heads_[h].projection.data(),
                       heads_[h].projection.size());
    }

    metrics_.avg_loss = total_loss / static_cast<float>(heads_.size());
    metrics_.step = step_;
    ++step_;
    return input_grad;
}

void MTPHeadTrainer::head_forward(const float* input, std::size_t batch_seq,
                                   std::size_t hidden, std::size_t head_idx,
                                   float* logits_out) {
    const auto& h = heads_[head_idx];
    const float* W = h.is_quantized ? h.quantized_projection.data() : h.projection.data();
    for (std::size_t i = 0; i < batch_seq; ++i) {
        for (std::size_t v = 0; v < cfg_.vocab_size; ++v) {
            float sum = h.output_bias[v];
            for (std::size_t j = 0; j < hidden; ++j) {
                sum += input[i * hidden + j] * W[j * cfg_.vocab_size + v];
            }
            logits_out[i * cfg_.vocab_size + v] = sum;
        }
    }
}

void MTPHeadTrainer::head_backward(const float* input, const float* grad_logits,
                                    std::size_t batch_seq, std::size_t hidden,
                                    std::size_t head_idx, float* input_grad_out) {
    const auto& h = heads_[head_idx];
    const float* W = h.is_quantized ? h.quantized_projection.data() : h.projection.data();

    std::size_t n = h.projection_rows * h.projection_cols;
    std::vector<float> w_grad(n, 0.0f);
    std::vector<float> b_grad(cfg_.vocab_size, 0.0f);

    for (std::size_t i = 0; i < batch_seq; ++i) {
        for (std::size_t v = 0; v < cfg_.vocab_size; ++v) {
            float g = grad_logits[i * cfg_.vocab_size + v];
            b_grad[v] += g;
            for (std::size_t j = 0; j < hidden; ++j) {
                w_grad[j * cfg_.vocab_size + v] += input[i * hidden + j] * g;
            }
        }
    }

    float scale = 1.0f / static_cast<float>(batch_seq);
    for (auto& g : w_grad) g *= scale;
    for (auto& g : b_grad) g *= scale;

    for (std::size_t i = 0; i < batch_seq; ++i) {
        for (std::size_t j = 0; j < hidden; ++j) {
            float sum = 0.0f;
            for (std::size_t v = 0; v < cfg_.vocab_size; ++v) {
                sum += W[j * cfg_.vocab_size + v] * grad_logits[i * cfg_.vocab_size + v];
            }
            input_grad_out[i * hidden + j] += sum;
        }
    }

    auto& momentum = momentum_[head_idx];
    auto& velocity = velocity_[head_idx];
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, lr = cfg_.learning_rate;
    std::size_t t = step_ + 1;

    for (std::size_t i = 0; i < n; ++i) {
        momentum[i] = beta1 * momentum[i] + (1.0f - beta1) * w_grad[i];
        velocity[i] = beta2 * velocity[i] + (1.0f - beta2) * w_grad[i] * w_grad[i];
        float m_hat = momentum[i] / (1.0f - std::pow(beta1, static_cast<float>(t)));
        float v_hat = velocity[i] / (1.0f - std::pow(beta2, static_cast<float>(t)));
        heads_[head_idx].projection[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) +
                                                 cfg_.weight_decay * heads_[head_idx].projection[i]);
    }
    for (std::size_t v = 0; v < cfg_.vocab_size; ++v) {
        heads_[head_idx].output_bias[v] -= lr * b_grad[v];
    }
}

void MTPHeadTrainer::update_weights(std::size_t head_idx, const float* grad, std::size_t n) {
    (void)head_idx;
    (void)grad;
    (void)n;
}

std::vector<std::vector<float>> MTPHeadTrainer::propose_tokens(
        const float* hidden_states, std::size_t batch_size,
        std::size_t seq_len, std::size_t hidden_size, std::size_t k) {
    std::size_t batch_seq = batch_size * seq_len;
    std::vector<std::vector<float>> proposals(batch_seq);

    for (std::size_t i = 0; i < batch_seq; ++i) {
        proposals[i].resize(k);
        std::vector<float> current_hidden(hidden_states + i * hidden_size,
                                           hidden_states + (i + 1) * hidden_size);

        for (std::size_t step = 0; step < k; ++step) {
            float best_val = -1e30f;
            std::size_t best_idx = 0;
            for (std::size_t h = 0; h < heads_.size(); ++h) {
                std::vector<float> logits(cfg_.vocab_size);
                head_forward(current_hidden.data(), 1, hidden_size, h, logits.data());
                for (std::size_t v = 0; v < cfg_.vocab_size; ++v) {
                    if (logits[v] > best_val) {
                        best_val = logits[v];
                        best_idx = v;
                    }
                }
            }
            proposals[i][step] = static_cast<float>(best_idx);
        }
    }
    return proposals;
}

void MTPHeadTrainer::quantize_heads(float target_bpw) {
    for (auto& h : heads_) {
        if (h.projection.empty()) continue;
        std::size_t n = h.projection.size();
        std::size_t k = static_cast<std::size_t>(1) << static_cast<int>(target_bpw);
        if (k < 2) k = 2;
        if (k > 256) k = 256;

        float mn = *std::min_element(h.projection.begin(), h.projection.end());
        float mx = *std::max_element(h.projection.begin(), h.projection.end());
        h.scale = (mx - mn > 1e-12f) ? (mx - mn) / static_cast<float>(k - 1) : 1.0f;
        if (h.scale < 1e-12f) h.scale = 1e-12f;

        h.quantized_projection.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            float normalized = (h.projection[i] - mn) / (h.scale * static_cast<float>(k - 1));
            float clamped = std::max(0.0f, std::min(static_cast<float>(k - 1), std::round(normalized)));
            h.quantized_projection[i] = mn + clamped * h.scale;
        }
        h.is_quantized = true;
    }
}

void MTPHeadTrainer::update_draft_k(float measured_acceptance_rate) {
    metrics_.acceptance_rate = measured_acceptance_rate;
    if (step_ % static_cast<std::size_t>(cfg_.auto_tune_window) != 0) return;

    if (measured_acceptance_rate > cfg_.target_acceptance_rate) {
        metrics_.current_draft_k = (std::min)(metrics_.current_draft_k + 1, cfg_.max_draft_k);
    } else if (measured_acceptance_rate < cfg_.target_acceptance_rate * 0.5f) {
        metrics_.current_draft_k = (std::max)(metrics_.current_draft_k - 1, cfg_.min_draft_k);
    }
}

} // namespace adapters
} // namespace oil
