#pragma once

#include "quant/tensor.h"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <functional>

namespace quant {

// ============================================================
// Optimizer base
// ============================================================
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad();
    void add_param(Tensor* param);
    void add_param_group(std::vector<Tensor*> params);

    struct ParamState {
        Tensor m, v, u;  // u for Adamax infinity norm, reused by others
    };

    float get_lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }
    float get_weight_decay() const { return weight_decay_; }
    void set_weight_decay(float wd) { weight_decay_ = wd; }
    int step_count() const { return t_; }

    // Gradient clipping
    void set_grad_clip_norm(float max_norm) { grad_clip_norm_ = max_norm; }
    float get_grad_clip_norm() const { return grad_clip_norm_; }
    float clip_grad_norm();

    ParamState& get_state(Tensor* p) { return state_[p]; }
    const std::unordered_map<Tensor*, ParamState>& all_state() const { return state_; }
    std::unordered_map<Tensor*, ParamState>& mutable_state() { return state_; }

protected:
    Optimizer(float lr, float weight_decay)
        : lr_(lr), weight_decay_(weight_decay) {}
    std::vector<Tensor*> parameters_;
    std::unordered_map<Tensor*, ParamState> state_;
    float lr_, weight_decay_;
    int t_ = 0;
    float grad_clip_norm_ = 0.0f; // 0 = no clipping
};

// ============================================================
// AdamW — decoupled weight decay + adaptive momentum
// ============================================================
class AdamW : public Optimizer {
public:
    AdamW(float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 1e-2f);
    void step() override;
    void zero_grad() override;
    void scheduler_step(int step);
    enum class Schedule { CONSTANT, COSINE, LINEAR, WARMUP_COSINE, COSINE_RESTART };
    void set_schedule(Schedule s, int warmup_steps = 0, int total_steps = 0,
                      int restart_period = 0);
private:
    float beta1_, beta2_, eps_;
    Schedule schedule_ = Schedule::CONSTANT;
    int warmup_steps_ = 0, total_steps_ = 0, restart_period_ = 0;
    float current_lr_ = 0;
    float adjusted_lr(int step);
};

// ============================================================
// SGD — with optional momentum and Nesterov acceleration
// ============================================================
class SGD : public Optimizer {
public:
    SGD(float lr = 1e-3f, float momentum = 0.9f, float weight_decay = 0.0f,
        bool nesterov = false);
    void step() override;
    void zero_grad() override;
private:
    float momentum_;
    bool nesterov_;
};

// ============================================================
// Adam — standard Adam (no weight decay decoupling)
// ============================================================
class Adam : public Optimizer {
public:
    Adam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    void zero_grad() override;
private:
    float beta1_, beta2_, eps_;
};

// ============================================================
// Adamax — Adam with L-infinity norm (stable for large embeddings)
// ============================================================
class Adamax : public Optimizer {
public:
    Adamax(float lr = 2e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
           float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    void zero_grad() override;
private:
    float beta1_, beta2_, eps_;
};

// ============================================================
// NAdam — Adam with Nesterov momentum
// ============================================================
class NAdam : public Optimizer {
public:
    NAdam(float lr = 2e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.0f, float momentum_decay = 0.004f);
    void step() override;
    void zero_grad() override;
private:
    float beta1_, beta2_, eps_, momentum_decay_;
};

// ============================================================
// RAdam — Rectified Adam (adaptive variance rectification)
// ============================================================
class RAdam : public Optimizer {
public:
    RAdam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    void zero_grad() override;
private:
    float beta1_, beta2_, eps_;
};

// ============================================================
// Lion — EvoLved Sign Momentum (Google, 2023)
// ============================================================
class Lion : public Optimizer {
public:
    Lion(float lr = 1e-4f, float beta1 = 0.9f, float beta2 = 0.99f,
         float weight_decay = 0.0f);
    void step() override;
    void zero_grad() override;
private:
    float beta1_, beta2_;
};

// ============================================================
// Adafactor — memory-efficient adaptive (factorized second moment)
// ============================================================
class Adafactor : public Optimizer {
public:
    Adafactor(float lr = 1e-2f, float beta2 = 0.999f, float eps = 1e-30f,
              float weight_decay = 0.0f, float clip_threshold = 1.0f,
              float decay_rate = 0.8f);
    void step() override;
    void zero_grad() override;
private:
    float beta2_, eps_, clip_threshold_, decay_rate_;
    struct FactorState {
        Tensor m; Tensor r; Tensor c; // mean, row, col
    };
    std::unordered_map<Tensor*, FactorState> factor_state_;
};

// ============================================================
// RMSProp — Root Mean Square Propagation
// ============================================================
class RMSProp : public Optimizer {
public:
    RMSProp(float lr = 1e-2f, float rho = 0.99f, float eps = 1e-8f,
            float weight_decay = 0.0f, float momentum = 0.0f);
    void step() override;
    void zero_grad() override;
private:
    float rho_, eps_, momentum_;
};

// ============================================================
// Ranger — RAdam + LookAhead wrapper
// ============================================================
class Ranger : public Optimizer {
public:
    Ranger(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
           float eps = 1e-8f, float weight_decay = 0.0f,
           int lookahead_k = 6, float lookahead_alpha = 0.5f);
    ~Ranger() override = default;
    void step() override;
    void zero_grad() override;
    void reset();
private:
    std::unique_ptr<RAdam> radam_;
    int lookahead_k_, slow_step_;
    float lookahead_alpha_;
    std::vector<Tensor> slow_weights_;
    void sync_slow_weights();
    void lookahead_step();
};

// ============================================================
// Adabound — Dynamic bounds on Adam learning rate
// ============================================================
class Adabound : public Optimizer {
public:
    Adabound(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
             float eps = 1e-8f, float weight_decay = 0.0f,
             float final_lr = 0.1f, float gamma = 1e-3f);
    void step() override;
    void zero_grad() override;
    void reset();
private:
    float beta1_, beta2_, eps_, final_lr_, gamma_;
};

// ============================================================
// Lamb — Trust ratio clipping, layer-wise adaptation
// ============================================================
class Lamb : public Optimizer {
public:
    Lamb(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f,
         float max_trust_ratio = 10.0f);
    void step() override;
    void zero_grad() override;
    void reset();
private:
    float beta1_, beta2_, eps_, max_trust_ratio_;
};

// ============================================================
// LARS — Layer-wise adaptive rate scaling
// ============================================================
class LARS : public Optimizer {
public:
    LARS(float lr = 1e-3f, float momentum = 0.9f, float weight_decay = 1e-4f,
         float trust_coefficient = 0.001f, float eps = 1e-8f);
    void step() override;
    void zero_grad() override;
    void reset();
private:
    float momentum_, trust_coefficient_, eps_;
};

// ============================================================
// NovoGrad — Normalized gradient with Adam-style moments
// ============================================================
class NovoGrad : public Optimizer {
public:
    NovoGrad(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
             float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    void zero_grad() override;
    void reset();
private:
    float beta1_, beta2_, eps_;
};

} // namespace quant
