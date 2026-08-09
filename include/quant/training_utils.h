#pragma once
#include "quant/tensor.h"
#include "quant/optimizer.h"
#include <vector>
#include <memory>

namespace quant {

// MoE auxiliary losses
struct MoEAuxLosses {
    Tensor load_balance_loss;  // scalar
    Tensor z_loss;             // scalar
    Tensor importance_loss;    // scalar
    float load_balance_weight = 0.01f;
    float z_loss_weight = 0.0001f;
    float importance_weight = 0.01f;
};

MoEAuxLosses compute_moe_aux_losses(
    const Tensor& router_logits,
    const Tensor& expert_gates,
    const Tensor& expert_mask,
    const MoEAuxLosses& cfg = MoEAuxLosses());

Tensor compute_load_balance_loss(const Tensor& expert_mask);
Tensor compute_z_loss(const Tensor& router_logits);
Tensor compute_importance_loss(const Tensor& expert_gates);

// Mixed precision with dynamic loss scaling
class MixedPrecisionScaler {
public:
    explicit MixedPrecisionScaler(float init_scale = 65536.0f,
                                    float growth_factor = 2.0f,
                                    float backoff_factor = 0.5f,
                                    int growth_interval = 2000);

    Tensor scale_loss(const Tensor& loss);
    bool check_gradients(const std::vector<Tensor*>& params_grad);
    void update_scale(bool has_overflow);

    float current_scale() const { return scale_; }
    int good_steps() const { return good_steps_; }
    void reset();

private:
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    int good_steps_;
};

// ===========================================================================
// MixedPrecisionManager — FP16/FP32 mixed precision with master weights
// ===========================================================================
class MixedPrecisionManager {
public:
    explicit MixedPrecisionManager(bool enabled = true,
                                    float init_loss_scale = 65536.0f,
                                    float growth_factor = 2.0f,
                                    float backoff_factor = 0.5f,
                                    int growth_interval = 2000);

    // Convert tensor between FP16 and FP32
    static Tensor to_fp16(const Tensor& t);
    static Tensor to_fp32(const Tensor& t);

    // Master weight management
    void create_master_weights(const std::vector<Tensor*>& model_params);
    void sync_weights_to_model(std::vector<Tensor*>& model_params);
    void sync_weights_from_model(const std::vector<Tensor*>& model_params);

    // Loss scaling
    Tensor scale_loss(const Tensor& loss);
    bool check_gradients(const std::vector<Tensor*>& grads);
    void unscale_gradients(const std::vector<Tensor*>& params);
    void update_loss_scale(bool has_overflow);

    // Cast gradients to FP32 before optimizer step
    void cast_gradients_to_fp32(const std::vector<Tensor*>& params);

    bool is_enabled() const { return enabled_; }
    float current_loss_scale() const { return loss_scale_; }

    void reset();

private:
    bool enabled_;
    float loss_scale_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    int good_steps_;
    std::vector<Tensor> master_weights_;

    static void fp32_to_fp16_bulk(const float* src, uint16_t* dst, int64_t n);
    static void fp16_to_fp32_bulk(const uint16_t* src, float* dst, int64_t n);
};

// Exponential Moving Average (EMA) for weights
class EMAWeightAveraging {
public:
    explicit EMAWeightAveraging(float decay = 0.999f, bool apply_ema = true);

    void init(const std::vector<Tensor*>& params);
    void update(const std::vector<Tensor*>& params);
    void apply_to_model(std::vector<Tensor*>& model_params);
    void copy_to_model(std::vector<Tensor*>& model_params);

private:
    float decay_;
    bool apply_ema_;
    std::vector<Tensor> ema_weights_;
    int64_t step_;
};

// ============================================================
// ProgressTracker
// ============================================================
class ProgressTracker {
public:
    ProgressTracker(int total_steps, int log_interval = 100);
    void step();
    void reset();
    bool is_done() const { return current_step_ >= total_steps_; }
    int current_step() const { return current_step_; }
    int total_steps() const { return total_steps_; }
    float progress() const;
    double elapsed_sec() const;
    double eta_sec() const;
private:
    int total_steps_, current_step_, log_interval_;
    double start_time_;
};

// ============================================================
// MetricTracker
// ============================================================
class MetricTracker {
public:
    MetricTracker(float alpha = 0.9f, bool minimize = true);
    void update(float value);
    void reset();
    float value() const { return running_avg_; }
    float best() const { return best_; }
    int steps() const { return steps_; }
    bool improved() const;
    float alpha() const { return alpha_; }
private:
    float alpha_, running_avg_, best_;
    int steps_;
    bool minimize_;
};

// ============================================================
// ModelSnapshot
// ============================================================
class ModelSnapshot {
public:
    explicit ModelSnapshot(const std::string& path = "snapshot");
    void save(Optimizer* opt, int step, float current_lr,
              const std::vector<Tensor*>& params);
    bool load(Optimizer* opt, int& step, float& current_lr,
              const std::vector<Tensor*>& params);
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

// ============================================================
// EarlyStopping
// ============================================================
class EarlyStopping {
public:
    EarlyStopping(int patience = 5, float min_delta = 1e-4f, bool minimize = true);
    bool step(float metric);
    void reset();
    bool stopped() const { return stopped_; }
    int best_epoch() const { return best_epoch_; }
    float best_metric() const { return best_metric_; }
    int patience() const { return patience_; }
private:
    int patience_, best_epoch_, bad_epochs_;
    float min_delta_, best_metric_;
    bool minimize_, stopped_;
};

// ============================================================
// LearningRateWarmup
// ============================================================
class LearningRateWarmup {
public:
    enum class Mode : uint8_t { LINEAR, EXPONENTIAL };
    LearningRateWarmup(float target_lr, int warmup_steps,
                       Mode mode = Mode::LINEAR);
    float get_lr(int step) const;
    int warmup_steps() const { return warmup_steps_; }
    float target_lr() const { return target_lr_; }
private:
    float target_lr_;
    int warmup_steps_;
    Mode mode_;
};

// ============================================================
// GradientAccumulator
// ============================================================
class GradientAccumulator {
public:
    explicit GradientAccumulator(int accumulation_steps = 1);
    void accumulate(const std::vector<Tensor*>& params);
    void apply_grads(const std::vector<Tensor*>& params, Optimizer* opt = nullptr);
    void reset();
    int current_step() const { return current_step_; }
    int accumulation_steps() const { return accumulation_steps_; }
    bool ready() const { return current_step_ >= accumulation_steps_; }
private:
    int accumulation_steps_, current_step_;
    std::vector<Tensor> grad_buffer_;
    bool initialized_ = false;
};

} // namespace quant
