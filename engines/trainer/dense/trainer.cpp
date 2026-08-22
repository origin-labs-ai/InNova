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
        AutogradEngine::instance().register_parameter(p);
        optimizer_.add_param(p);
    }

    qat_enabled_ = cfg.use_qat;
    qat_bits_ = cfg.qat_bits;
    qat_symmetric_ = cfg.qat_symmetric;
    qat_use_lsq_ = cfg.qat_use_lsq;
    qat_init_scale_ = cfg.qat_init_scale;
    qat_scales_.clear();

    step_ = 0;
    metrics_ = TrainMetrics{};
}

float DenseTrainer::train_step(const Tensor& input_ids, const Tensor& labels) {
    zero_grad();
    AutogradEngine::instance().clear();
    const bool prev_ag = AutogradEngine::enabled();
    AutogradEngine::set_enabled(true);

    // QAT: inject FakeQuantize into weights before forward — STE grad passthrough
    std::vector<std::vector<float>> qat_backup;
    auto params_qat = get_parameters();
    if (qat_enabled_) {
        bool need_init = qat_scales_.size() != params_qat.size();
        if (need_init) {
            qat_scales_.clear();
            for (auto* p : params_qat) {
                int qmin, qmax;
                qat::get_qrange_bits(qat_bits_, qat_symmetric_, false, qmin, qmax);
                const float* d = p->data<float>();
                float mx = 0; int64_t n = p->numel();
                for (int64_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(d[i]));
                if (mx < 1e-8f) mx = 1.0f;
                float sc = mx / (float)qmax;
                if (sc < 1e-6f) sc = qat_init_scale_;
                Tensor ts(Shape{1}, DType::F32);
                ts.data<float>()[0] = sc;
                ts.requires_grad(qat_use_lsq_);
                qat_scales_.push_back(ts);
            }
            if (qat_use_lsq_) {
                for (size_t i = 0; i < qat_scales_.size(); ++i) {
                    AutogradEngine::instance().register_parameter(&qat_scales_[i]);
                    optimizer_.add_param(&qat_scales_[i]);
                }
            }
        } else if (qat_use_lsq_) {
            for (size_t i = 0; i < qat_scales_.size(); ++i) AutogradEngine::instance().register_parameter(&qat_scales_[i]);
        }
        qat_backup.resize(params_qat.size());
        for (size_t i = 0; i < params_qat.size(); ++i) {
            qat_backup[i].assign(params_qat[i]->data<float>(), params_qat[i]->data<float>() + params_qat[i]->numel());
            int qmin, qmax;
            qat::get_qrange_bits(qat_bits_, qat_symmetric_, false, qmin, qmax);
            Tensor q = qat_use_lsq_ ? qat::lsq_fake_quantize(*params_qat[i], qat_scales_[i], qmin, qmax)
                                     : qat::fake_quantize(*params_qat[i], qat_scales_[i].data<float>()[0], qmin, qmax);
            std::memcpy(params_qat[i]->data<float>(), q.data<float>(), (size_t)params_qat[i]->numel() * sizeof(float));
        }
    }

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

    // Continual replay mixing (second loss term)
    Tensor replay_loss;
    bool has_replay = false;
    if (continual_enabled_ && continual_engine_ && continual_engine_->replay().size() > 0 && replay_ratio_ > 0.0f) {
        std::vector<std::vector<float>> r_inputs, r_targets;
        std::vector<float> r_weights;
        size_t rb = (size_t)std::min<int64_t>(B, 4);
        if (continual_engine_->replay().sample_batch(r_inputs, r_targets, r_weights, rb)) {
            Tensor r_in(Shape{(int64_t)rb, T});
            Tensor r_lab(Shape{(int64_t)rb, T});
            float* ri = r_in.data<float>(); float* rl = r_lab.data<float>();
            for (size_t b = 0; b < rb; b++) {
                size_t cn = std::min<size_t>((size_t)T, r_inputs[b].size());
                for (int64_t s = 0; s < T; s++) {
                    ri[b*T+s] = s < (int64_t)cn ? r_inputs[b][s] : 0.0f;
                    rl[b*T+s] = s < (int64_t)cn && s < (int64_t)r_targets[b].size() ? r_targets[b][s] : ri[b*T+s];
                }
            }
            Tensor r_pos(Shape{(int64_t)rb, T});
            float* rp = r_pos.data<float>(); for (int64_t i=0;i<(int64_t)rb*T;i++) rp[i]=(float)(i % T);
            Tensor r_logits = model_->forward(r_in, r_pos);
            int64_t RBT = (int64_t)rb * T;
            Tensor r_flat = r_logits.reshape({RBT, cfg.vocab_size});
            Tensor rl_flat = r_lab.reshape({RBT});
            replay_loss = AutogradEngine::cross_entropy_op(r_flat, rl_flat);
            has_replay = true;
        }
    }

    AutogradEngine::instance().backward(loss_tensor);
    if (has_replay) {
        std::vector<std::vector<float>> before;
        auto params = get_parameters();
        before.reserve(params.size());
        for (auto* p : params) {
            if (p->has_grad()) { float* gd=p->grad().data<float>(); before.emplace_back(gd, gd+p->grad().numel()); }
            else before.emplace_back();
        }
        AutogradEngine::instance().backward(replay_loss);
        auto params2 = get_parameters();
        for (size_t i=0;i<params2.size();i++) {
            auto* p=params2[i];
            if (!p->has_grad() || before[i].empty()) continue;
            float* gd=p->grad().data<float>();
            for (int64_t k=0;k<p->grad().numel();k++) {
                float delta=gd[k]-before[i][k];
                gd[k]=before[i][k]+delta*replay_ratio_;
            }
        }
        loss_val += replay_ratio_ * replay_loss.data<float>()[0];
    }
    // EWC regularization gradient
    if (continual_enabled_ && continual_engine_ && continual_engine_->ecc().initialized) {
        const auto& ecc = continual_engine_->ecc();
        size_t off=0;
        auto params = get_parameters();
        for (auto* p : params) {
            if (!p->has_grad()) { off+=(size_t)p->numel(); continue; }
            float* gd=p->grad().data<float>();
            const float* wd=p->data<float>();
            int64_t n=p->numel();
            for (int64_t i=0;i<n;i++) {
                size_t idx=off+(size_t)i;
                if (idx < ecc.fisher_diagonal.size() && idx < ecc.anchor_weights.size()) {
                    float diff=wd[i]-ecc.anchor_weights[idx];
                    gd[i]+=2.0f*ewc_lambda_*ecc.fisher_diagonal[idx]*diff;
                }
            }
            off+=(size_t)n;
        }
        // add scalar to reported loss
        size_t total=0; auto pr=get_parameters(); for(auto* p:pr) total+=(size_t)p->numel();
        std::vector<float> flat; flat.reserve(total);
        for(auto* p:pr){ const float* d=p->data<float>(); flat.insert(flat.end(), d, d+p->numel()); }
        if (!flat.empty()) loss_val += ecc.regularize(flat.data(), flat.size(), ewc_lambda_);
    }
    if (qat_enabled_ && !qat_backup.empty()) {
        for (size_t i = 0; i < params_qat.size() && i < qat_backup.size(); ++i)
            std::memcpy(params_qat[i]->data<float>(), qat_backup[i].data(), qat_backup[i].size() * sizeof(float));
    }

    // Update Fisher and insert replay
    if (continual_enabled_ && continual_engine_) {
        auto params = get_parameters();
        size_t total=0; for(auto* p:params) total+=(size_t)p->numel();
        std::vector<float> flat_g; flat_g.reserve(total);
        for(auto* p:params) if(p->has_grad()){ const float* gd=p->grad().data<float>(); flat_g.insert(flat_g.end(), gd, gd+p->grad().numel()); } else flat_g.insert(flat_g.end(), (size_t)p->numel(), 0.0f);
        std::vector<float> flat_w; flat_w.reserve(total);
        for(auto* p:params){ const float* d=p->data<float>(); flat_w.insert(flat_w.end(), d, d+p->numel()); }
        if (!flat_g.empty() && !flat_w.empty()) {
            size_t n=std::min(flat_g.size(), flat_w.size());
            continual_engine_->on_step(flat_g.data(), n, flat_w.data(), n, optimizer_.get_lr(), current_task_id_);
        }
        std::vector<float> fin, flab;
        fin.reserve((size_t)(B*T)); flab.reserve((size_t)(B*T));
        const float* id=input_ids.data<float>(); const float* lb=labels.data<float>();
        for(int64_t i=0;i<B*T;i++){ fin.push_back(id[i]); flab.push_back(lb[i]); }
        float imp=1.0f; if(!flat_g.empty()){ double sq=0; for(float g:flat_g) sq+=(double)g*g; imp=(float)std::sqrt(sq)+0.01f; }
        if(!fin.empty()) continual_engine_->replay().insert(fin.data(), flab.data(), fin.size(), current_task_id_, imp);
    }

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

void DenseTrainer::enable_continual(const ContinualEngineConfig& cfg) {
    continual_cfg_=cfg; ewc_lambda_=cfg.ecc_lambda; continual_engine_=std::make_unique<ContinualEngine>(cfg); continual_enabled_=true; current_task_id_=0;
}
void DenseTrainer::disable_continual(){ continual_enabled_=false; continual_engine_.reset(); }
void DenseTrainer::on_task_boundary(uint32_t nid, const float* ei, const float* et, size_t ec){
    current_task_id_=nid;
    if(continual_engine_){
        auto params=get_parameters();
        size_t total=0; for(auto* p:params) total+=(size_t)p->numel();
        std::vector<float> flat; flat.reserve(total);
        for(auto* p:params){ const float* d=p->data<float>(); flat.insert(flat.end(), d, d+p->numel()); }
        if(!flat.empty()){
            if(!continual_engine_->ecc().initialized) const_cast<ECCState&>(continual_engine_->ecc()).initialize(flat.size());
            const_cast<ECCState&>(continual_engine_->ecc()).update_anchor(flat.data(), flat.size());
        }
        continual_engine_->on_task_boundary(nid, ei, et, ec);
    }
}

void DenseTrainer::enable_qat(int bits, bool symmetric, bool use_lsq, float init_scale) {
    qat_enabled_ = true; qat_bits_ = bits; qat_symmetric_ = symmetric; qat_use_lsq_ = use_lsq; qat_init_scale_ = init_scale; qat_scales_.clear();
}
void DenseTrainer::disable_qat() { qat_enabled_ = false; qat_scales_.clear(); }
Tensor DenseTrainer::qat_fake_quantize(const Tensor& w) {
    int qmin, qmax; qat::get_qrange_bits(qat_bits_, qat_symmetric_, false, qmin, qmax);
    const float* d = w.data<float>(); float mx = 0; int64_t n = w.numel();
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(d[i]));
    if (mx < 1e-8f) mx = 1.0f; float sc = mx / (float)qmax;
    return qat::fake_quantize(w, sc, qmin, qmax);
}
Tensor DenseTrainer::qat_fake_quantize_lsq(const Tensor& w, Tensor& sp) {
    int qmin, qmax; qat::get_qrange_bits(qat_bits_, qat_symmetric_, false, qmin, qmax);
    return qat::lsq_fake_quantize(w, sp, qmin, qmax);
}

} // namespace dense
} // namespace quant
