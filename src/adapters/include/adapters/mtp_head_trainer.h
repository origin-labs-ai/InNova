#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <random>
#include "quant/types.h"
#include "quant/format_registry.h"

namespace quant {
namespace adapters {

struct MTPHeadConfig {
    std::size_t num_heads = 3;
    std::size_t hidden_size = 768;
    std::size_t vocab_size = 50257;
    float learning_rate = 1e-4f;
    float weight_decay = 1e-2f;
    float draft_quantize_bpw = 2.0f;
    bool quantize_draft_heads = true;
    int auto_tune_window = 100;
    float target_acceptance_rate = 0.7f;
    int min_draft_k = 1;
    int max_draft_k = 8;
};

struct MTPHeadMetrics {
    std::vector<float> head_losses;
    float avg_loss = 0.0f;
    float acceptance_rate = 0.0f;
    int current_draft_k = 3;
    std::size_t total_tokens_accepted = 0;
    std::size_t total_tokens_proposed = 0;
    std::size_t step = 0;
};

struct MTPHeadWeights {
    std::vector<float> projection;
    std::vector<float> output_bias;
    std::vector<float> quantized_projection;
    float scale = 1.0f;
    bool is_quantized = false;
    std::size_t projection_rows = 0;
    std::size_t projection_cols = 0;
};

class MTPHeadTrainer {
public:
    explicit MTPHeadTrainer(const MTPHeadConfig& cfg = {});
    ~MTPHeadTrainer() = default;

    void initialize();

    std::vector<float> train_step(const float* hidden_states, const float* targets,
                                  std::size_t batch_size, std::size_t seq_len,
                                  std::size_t hidden_size);

    std::vector<std::vector<float>> propose_tokens(const float* hidden_states,
                                                    std::size_t batch_size,
                                                    std::size_t seq_len,
                                                    std::size_t hidden_size,
                                                    std::size_t k);

    void quantize_heads(float target_bpw);

    void update_draft_k(float measured_acceptance_rate);

    [[nodiscard]] int current_draft_k() const { return metrics_.current_draft_k; }
    [[nodiscard]] const MTPHeadMetrics& metrics() const { return metrics_; }
    [[nodiscard]] const MTPHeadWeights& head(std::size_t i) const { return heads_.at(i); }
    [[nodiscard]] std::size_t num_heads() const { return heads_.size(); }

private:
    void head_forward(const float* input, std::size_t batch_seq, std::size_t hidden,
                      std::size_t head_idx, float* logits_out);
    void head_backward(const float* input, const float* grad_logits,
                       std::size_t batch_seq, std::size_t hidden, std::size_t head_idx,
                       float* input_grad_out);
    void update_weights(std::size_t head_idx, const float* grad, std::size_t n);

    MTPHeadConfig cfg_;
    MTPHeadMetrics metrics_;
    std::vector<MTPHeadWeights> heads_;
    // Adafactor optimizer state (per head):
    //   momentum_     — first moment m_t (beta1=0.9)
    //   row_sq_       — row-wise second moment  R_t (size hidden)
    //   col_sq_       — column-wise second moment C_t (size vocab)
    //   bias_sq_      — second moment for the output bias (size vocab)
    std::vector<std::vector<float>> momentum_;
    std::vector<std::vector<float>> row_sq_;
    std::vector<std::vector<float>> col_sq_;
    std::vector<std::vector<float>> bias_sq_;
    // Gradient for the current head's projection/bias, stashed by
    // head_backward and consumed by update_weights (real Adafactor update).
    std::vector<float> pending_w_grad_;
    std::vector<float> pending_b_grad_;
    std::size_t step_ = 0;
    std::mt19937_64 rng_{42};

    // Gradients computed by head_backward, consumed by update_weights so the
    // AdamW step is a real, separate operation.
    std::vector<float> pending_proj_grad_;
    std::vector<float> pending_bias_grad_;
};

} // namespace adapters
} // namespace quant
