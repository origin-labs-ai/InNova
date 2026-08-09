#include "trainer.h"
#include "quant/math.h"
#include <cmath>
#include <chrono>
#include <iostream>

namespace quant {
namespace dense {

DenseTrainer::DenseTrainer(DenseModel* model, Tokenizer* tokenizer)
    : model_(model), tokenizer_(tokenizer),
      optimizer_(3e-4f, 0.9f, 0.999f, 1e-8f, 1e-2f)
{
}

std::vector<Tensor*> DenseTrainer::get_parameters() {
    std::vector<Tensor*> params;

    auto add_tensor = [&](Tensor* t) {
        if (t && t->numel() > 0) params.push_back(t);
    };

    if (model_->tok_embeddings)
        add_tensor(&model_->tok_embeddings->weight);

    for (auto& layer : model_->layers) {
        add_tensor(&layer->attention_norm.weight);
        add_tensor(&layer->attention.q_proj.weight);
        add_tensor(&layer->attention.q_proj.bias);
        add_tensor(&layer->attention.k_proj.weight);
        add_tensor(&layer->attention.k_proj.bias);
        add_tensor(&layer->attention.v_proj.weight);
        add_tensor(&layer->attention.v_proj.bias);
        add_tensor(&layer->attention.o_proj.weight);
        add_tensor(&layer->attention.o_proj.bias);
        add_tensor(&layer->ffn_norm.weight);
        add_tensor(&layer->ffn.gate_proj.weight);
        add_tensor(&layer->ffn.gate_proj.bias);
        add_tensor(&layer->ffn.up_proj.weight);
        add_tensor(&layer->ffn.up_proj.bias);
        add_tensor(&layer->ffn.down_proj.weight);
        add_tensor(&layer->ffn.down_proj.bias);
    }

    if (model_->norm)
        add_tensor(&model_->norm->weight);

    if (model_->lm_head) {
        add_tensor(&model_->lm_head->weight);
        add_tensor(&model_->lm_head->bias);
    }

    return params;
}

void DenseTrainer::compile(const TrainConfig& cfg) {
    config_ = cfg;

    optimizer_ = AdamW(cfg.learning_rate, 0.9f, 0.999f, 1e-8f, cfg.weight_decay);

    auto params = get_parameters();
    for (auto* p : params) {
        p->requires_grad(true);
        optimizer_.add_param(p);
    }

    step_ = 0;
    metrics_ = TrainMetrics{};
}

float DenseTrainer::train_step(const Tensor& input_ids, const Tensor& labels) {
    zero_grad();
    AutogradEngine::instance().clear();
    const bool prev_ag = AutogradEngine::enabled();
    AutogradEngine::set_enabled(true);

    auto cfg = model_->config;
    int64_t B = input_ids.dim(0);
    int64_t T = input_ids.dim(1);

    Tensor positions = Tensor(Shape(T));
    float* pos_data = positions.data<float>();
    for (int64_t i = 0; i < T; ++i)
        pos_data[i] = (float)i;

    Tensor logits = model_->forward(input_ids, positions);

    int64_t BT = B * T;
    Tensor logits_flat = logits.reshape({BT, cfg.vocab_size});
    Tensor labels_flat = labels.reshape({BT});

    // Use autograd-aware CE so the graph reaches parameters
    Tensor loss_tensor = AutogradEngine::cross_entropy_op(logits_flat, labels_flat);
    float loss_val = loss_tensor.data<float>()[0];

    AutogradEngine::instance().backward(loss_tensor);

    AutogradEngine::set_enabled(prev_ag);

    clip_gradients(config_.grad_clip);

    optimizer_.step();

    metrics_.loss = loss_val;
    metrics_.perplexity = std::exp(loss_val);
    metrics_.learning_rate = optimizer_.get_lr();
    metrics_.step = ++step_;

    return loss_val;
}

void DenseTrainer::zero_grad() {
    optimizer_.zero_grad();
}

void DenseTrainer::clip_gradients(float max_norm) {
    if (max_norm <= 0.0f) {
        metrics_.grad_norm = 0.0f;
        return;
    }

    auto params = get_parameters();
    float total_norm = 0.0f;

    for (auto* p : params) {
        if (!p->requires_grad()) continue;
        const Tensor& g_ref = p->grad();
        if (g_ref.numel() == 0) continue;
        float gn = math::norm(g_ref);
        total_norm += gn * gn;
    }

    total_norm = std::sqrt(total_norm);
    metrics_.grad_norm = total_norm;

    if (total_norm > max_norm && total_norm > 0.0f) {
        float scale = max_norm / total_norm;
        for (auto* p : params) {
            if (!p->requires_grad()) continue;
            Tensor g = p->grad();
            float* gd = g.data<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i)
                gd[i] *= scale;
            p->set_grad(g);
        }
    }
}

void DenseTrainer::fit(DataLoader& loader) {
    int64_t total_steps = (int64_t)config_.num_epochs * loader.num_batches();
    optimizer_.set_schedule(AdamW::Schedule::WARMUP_COSINE, config_.warmup_steps, (int)total_steps);

    auto start_time = std::chrono::steady_clock::now();
    int64_t total_tokens = 0;

    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        if (epoch > 0) loader.shuffle();
        loader.reset();

        int batch_idx = 0;
        Tensor input_ids, labels;

        while (loader.next_batch(input_ids, labels)) {
            float loss = train_step(input_ids, labels);

            int64_t tokens = input_ids.numel();
            total_tokens += tokens;

            metrics_.epoch_progress = (float)(batch_idx + 1) / (float)loader.num_batches();

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            metrics_.tokens_per_sec = elapsed > 0 ? (int)(total_tokens / elapsed) : 0;

            if (step_ % config_.log_interval == 0) {
                if (log_cb_)
                    log_cb_(metrics_);
                else {
                    std::cout << "Step " << step_
                              << " | Epoch " << (epoch + 1) << "/" << config_.num_epochs
                              << " | Loss " << metrics_.loss
                              << " | PPL " << metrics_.perplexity
                              << " | LR " << metrics_.learning_rate
                              << " | GN " << metrics_.grad_norm
                              << " | tok/s " << metrics_.tokens_per_sec
                              << std::endl;
                }
            }

            if (step_ % config_.save_interval == 0) {
                std::string cp_path = config_.output_path + ".step" + std::to_string(step_);
                save_checkpoint(cp_path);
            }

            ++batch_idx;
        }
    }

    save_checkpoint(config_.output_path);

    if (log_cb_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        metrics_.tokens_per_sec = elapsed > 0 ? (int)(total_tokens / elapsed) : 0;
        log_cb_(metrics_);
    }
}

const TrainMetrics& DenseTrainer::metrics() const {
    return metrics_;
}

void DenseTrainer::set_log_callback(LogCallback cb) {
    log_cb_ = std::move(cb);
}

} // namespace dense
} // namespace quant
