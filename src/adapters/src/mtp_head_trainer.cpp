#include "adapters/mtp_head_trainer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace quant {
namespace adapters {

MTPHeadTrainer::MTPHeadTrainer(const MTPHeadConfig& cfg)
    : cfg_(cfg) {}

void MTPHeadTrainer::initialize() {
    heads_.clear();
    momentum_.clear();
    row_sq_.clear();
    col_sq_.clear();
    bias_sq_.clear();
    heads_.resize(cfg_.num_heads);
    momentum_.resize(cfg_.num_heads);
    row_sq_.resize(cfg_.num_heads);
    col_sq_.resize(cfg_.num_heads);
    bias_sq_.resize(cfg_.num_heads);

    for (std::size_t h = 0; h < cfg_.num_heads; ++h) {
        auto& head = heads_[h];
        // Projection maps the hidden state to vocab logits: shape hidden x vocab
        // (head_forward indexes it as W[j * vocab_size + v]).
        head.projection_rows = cfg_.hidden_size;
        head.projection_cols = cfg_.vocab_size;
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

        // Adafactor state: zero first/second moments.
        momentum_[h].assign(n, 0.0f);
        row_sq_[h].assign(cfg_.hidden_size, 0.0f);
        col_sq_[h].assign(cfg_.vocab_size, 0.0f);
        bias_sq_[h].assign(cfg_.vocab_size, 0.0f);
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
        update_weights(h, nullptr, heads_[h].projection.size());
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

    // Stash the real gradients for update_weights (Adafactor update).
    pending_w_grad_ = std::move(w_grad);
    pending_b_grad_ = std::move(b_grad);
}

void MTPHeadTrainer::update_weights(std::size_t head_idx, const float* grad, std::size_t n) {
    // REAL Adafactor update (Shazeer & Stern, 2018):
    //   beta2_t = 1 - t^-0.8
    //   2-D projection: factorized row/col second moments, v = R*C / mean(R)
    //   1-D bias:       plain second moment
    //   beta1 momentum in the numerator, updates clipped to RMS 1.0, and a
    //   relative step size scaled by parameter-RMS / update-RMS.
    if (head_idx >= heads_.size()) return;
    const float* g = grad ? grad : pending_w_grad_.data();
    if (!g) return;

    auto& head = heads_[head_idx];
    const std::size_t R = head.projection_rows;
    const std::size_t C = head.projection_cols;
    if (R * C != n || head.projection.size() != n) return;

    const std::size_t t = step_ + 1;
    const float beta2_t = 1.0f - std::pow((float)t, -0.8f);
    const float eps = 1e-8f;
    const float clip = 1.0f;
    const float min_rel = 1e-6f;
    const float beta1 = 0.9f;
    const float lr = cfg_.learning_rate;
    const float wd = cfg_.weight_decay;

    // --- Factorized second moments: R_t (row means) and C_t (col means) ---
    auto& row = row_sq_[head_idx];
    auto& col = col_sq_[head_idx];
    double row_sum = 0.0;
    for (std::size_t i = 0; i < R; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < C; ++j) {
            float gij = g[i * C + j];
            s += (double)gij * (double)gij;
        }
        row[i] = beta2_t * row[i] + (1.0f - beta2_t) * (float)(s / (double)C);
        row_sum += row[i];
    }
    for (std::size_t j = 0; j < C; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < R; ++i) {
            float gij = g[i * C + j];
            s += (double)gij * (double)gij;
        }
        s /= (double)R;
        col[j] = beta2_t * col[j] + (1.0f - beta2_t) * (float)s;
    }
    const float row_mean = (float)(row_sum / (double)std::max<std::size_t>(1, R));
    const float v_denom = std::max(row_mean, eps);

    // --- u = m / sqrt(v) with momentum ---
    std::vector<float> u(n);
    auto& mom = momentum_[head_idx];
    double update_sq = 0.0, param_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t r = i / C, c = i % C;
        float gi = g[i];
        mom[i] = beta1 * mom[i] + (1.0f - beta1) * gi;
        float v = (row[r] * col[c]) / v_denom;
        float ui = mom[i] / (std::sqrt(v) + eps);
        u[i] = ui;
        update_sq += (double)ui * (double)ui;
        param_sq += (double)head.projection[i] * (double)head.projection[i];
    }
    float param_rms = std::sqrt((float)(param_sq / (double)std::max<std::size_t>(1, n)));
    float update_rms = std::sqrt((float)(update_sq / (double)std::max<std::size_t>(1, n)));
    if (update_rms > clip) {
        float s = clip / update_rms;
        for (auto& ui : u) ui *= s;
        update_rms = clip;
    }
    float step_size = std::max(min_rel, lr * (param_rms / (update_rms + eps)));
    for (std::size_t i = 0; i < n; ++i) {
        head.projection[i] *= (1.0f - lr * wd);          // decoupled weight decay
        head.projection[i] -= step_size * u[i];
    }

    // --- Bias: plain second moment + same Adafactor recipe ---
    if (!pending_b_grad_.empty()) {
        auto& bs = bias_sq_[head_idx];
        auto& b = head.output_bias;
        const std::size_t V = std::min(pending_b_grad_.size(), b.size());
        std::vector<float> ub(V);
        double bu_sq = 0.0, bp_sq = 0.0;
        for (std::size_t j = 0; j < V; ++j) {
            float gb = pending_b_grad_[j];
            bs[j] = beta2_t * bs[j] + (1.0f - beta2_t) * gb * gb;
            float ubj = gb / (std::sqrt(bs[j]) + eps);
            ub[j] = ubj;
            bu_sq += (double)ubj * (double)ubj;
            bp_sq += (double)b[j] * (double)b[j];
        }
        float bp_rms = std::sqrt((float)(bp_sq / (double)std::max<std::size_t>(1, V)));
        float bu_rms = std::sqrt((float)(bu_sq / (double)std::max<std::size_t>(1, V)));
        if (bu_rms > clip) {
            float s = clip / bu_rms;
            for (auto& ui : ub) ui *= s;
            bu_rms = clip;
        }
        float bs_step = std::max(min_rel, lr * (bp_rms / (bu_rms + eps)));
        for (std::size_t j = 0; j < V; ++j) {
            b[j] *= (1.0f - lr * wd);
            b[j] -= bs_step * ub[j];
        }
    }
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
} // namespace quant
