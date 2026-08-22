#pragma once
#include "quant/tensor.h"
#include "quant/autograd.h"
#include <vector>
#include <memory>
#include <cmath>
#include <mutex>
#include <algorithm>
#include <limits>

namespace quant {
namespace qat {

// bhai ye QAT config hai — bits / symmetric / LSQ sab yahin se control hota hai
struct QATConfig {
    int bits = 8;                 // quantization bits (1..8 typical)
    bool symmetric = true;        // symmetric vs asymmetric
    bool per_channel = false;     // per-channel vs per-tensor (abhi per-tensor impl)
    bool narrow_range = false;    // narrow range (skip -128 for int8)
    float init_scale = 1.0f;      // initial scale for LSQ / fake-quant
    bool use_lsq = false;         // LSQ enable karo to scale trainable
    float lsq_grad_scale = 1.0f;  // gradient scaling for LSQ stability
};

struct QuantParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
    int qmin = -128;
    int qmax = 127;
};

inline void get_qrange(const QATConfig& cfg, int& qmin, int& qmax) {
    if (cfg.symmetric) {
        if (cfg.narrow_range) {
            qmin = -(1 << (cfg.bits - 1)) + 1;
            qmax =  (1 << (cfg.bits - 1)) - 1;
        } else {
            qmin = -(1 << (cfg.bits - 1));
            qmax =  (1 << (cfg.bits - 1)) - 1;
        }
    } else {
        qmin = 0;
        qmax =  (1 << cfg.bits) - 1;
    }
}

inline void get_qrange_bits(int bits, bool symmetric, bool narrow, int& qmin, int& qmax) {
    if (symmetric) {
        if (narrow) { qmin = -(1 << (bits - 1)) + 1; qmax = (1 << (bits - 1)) - 1; }
        else { qmin = -(1 << (bits - 1)); qmax = (1 << (bits - 1)) - 1; }
    } else {
        qmin = 0; qmax = (1 << bits) - 1;
    }
}

enum class ObserverType { MinMax, MovingAverage, KLDivergence };

// ============================================================================
// FakeQuantize — STE node
// forward: x_q = clamp(round(x / scale), qmin, qmax) * scale
// backward: STE identity — grad bypass seedha pass (gradient unchanged)
// ============================================================================
class FakeQuantizeFunction : public AutogradFunction {
public:
    FakeQuantizeFunction(float scale, int qmin, int qmax)
        : scale_(scale), qmin_(qmin), qmax_(qmax) {}
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
    float scale() const { return scale_; }
private:
    float scale_;
    int qmin_, qmax_;
};

// ============================================================================
// LSQ (Learned Step Size Quantization) — Esser et al. 2020
// scale is trainable param; grad w.r.t scale uses LSQ formula
// forward: same fake-quant but scale comes from Tensor param
// backward: grad_x = STE(grad), grad_scale = sum grad * indicator
// ============================================================================
class LSQFunction : public AutogradFunction {
public:
    LSQFunction(int qmin, int qmax) : qmin_(qmin), qmax_(qmax) {}
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
private:
    int qmin_, qmax_;
};

// Convenience wrapper with trainable scale
class LSQQuantizer {
public:
    explicit LSQQuantizer(const QATConfig& cfg);
    // per-tensor fake-quant via LSQ (autograd aware)
    Tensor forward(const Tensor& x);
    // version where scale_param is an external Tensor* (for optimizer)
    Tensor forward(const Tensor& x, Tensor& scale_param);
    float scale() const { return scale_; }
    void set_scale(float s) { scale_ = std::max(s, 1e-6f); }
    float compute_scale_grad(const Tensor& x, const Tensor& grad_output) const;
    QuantParams quant_params() const;
    const QATConfig& config() const { return cfg_; }
    int qmin() const { return qmin_; }
    int qmax() const { return qmax_; }
    // expose scale as Tensor param for optimizer registration
    Tensor scale_param_tensor();
private:
    QATConfig cfg_;
    float scale_;
    int qmin_, qmax_;
    void init_qrange();
};

// ============================================================================
// Observer base — calibration hooks
// ============================================================================
class Observer {
public:
    virtual ~Observer() = default;
    virtual void observe(const Tensor& data) = 0;
    virtual QuantParams calc_qparams(int bits, bool symmetric) const = 0;
    virtual void reset() = 0;
    virtual ObserverType type() const = 0;
    // helper: observe + calc in one shot
    QuantParams calibrate(const Tensor& data, int bits, bool symmetric) {
        observe(data);
        return calc_qparams(bits, symmetric);
    }
};

// MinMaxObserver — tracks global min/max
class MinMaxObserver : public Observer {
public:
    MinMaxObserver() : min_val_(std::numeric_limits<float>::max()),
                       max_val_(std::numeric_limits<float>::lowest()), has_data_(false) {}
    void observe(const Tensor& data) override;
    QuantParams calc_qparams(int bits, bool symmetric) const override;
    void reset() override { min_val_ = std::numeric_limits<float>::max(); max_val_ = std::numeric_limits<float>::lowest(); has_data_ = false; }
    ObserverType type() const override { return ObserverType::MinMax; }
    float min_val() const { return min_val_; }
    float max_val() const { return max_val_; }
private:
    float min_val_, max_val_;
    bool has_data_;
};

// MovingAverageMinMaxObserver — EMA of min/max (calibration hook for streaming)
class MovingAverageMinMaxObserver : public Observer {
public:
    explicit MovingAverageMinMaxObserver(float momentum = 0.01f)
        : momentum_(momentum), min_val_(0), max_val_(0), has_data_(false) {}
    void observe(const Tensor& data) override;
    QuantParams calc_qparams(int bits, bool symmetric) const override;
    void reset() override { min_val_ = 0; max_val_ = 0; has_data_ = false; }
    ObserverType type() const override { return ObserverType::MovingAverage; }
    float min_val() const { return min_val_; }
    float max_val() const { return max_val_; }
private:
    float momentum_;
    float min_val_, max_val_;
    bool has_data_;
};

// KLDivergenceObserver — histogram + KL to find optimal threshold
// (simplified TensorRT-style KL calibration; falls back to MinMax if degenerate)
class KLDivergenceObserver : public Observer {
public:
    explicit KLDivergenceObserver(int bins = 2048, int quantized_bins = 128)
        : bins_(bins), quantized_bins_(quantized_bins), has_data_(false),
          hist_min_(0), hist_max_(0) { histogram_.assign(bins_, 0); }
    void observe(const Tensor& data) override;
    QuantParams calc_qparams(int bits, bool symmetric) const override;
    void reset() override { std::fill(histogram_.begin(), histogram_.end(), 0); has_data_ = false; hist_min_=0; hist_max_=0; }
    ObserverType type() const override { return ObserverType::KLDivergence; }
private:
    int bins_;
    int quantized_bins_;
    bool has_data_;
    float hist_min_, hist_max_;
    std::vector<int64_t> histogram_;
    float compute_kl_threshold(int bits, bool symmetric) const;
};

// ============================================================================
// Functional helpers (autograd-aware; use STE when enabled)
// ============================================================================
Tensor fake_quantize(const Tensor& x, float scale, int qmin, int qmax);
Tensor fake_quantize_per_tensor(const Tensor& x, float scale, int bits, bool symmetric);
Tensor fake_quantize_with_observer(const Tensor& x, Observer& obs, int bits, bool symmetric);
// STE passthrough — gradient seedha pass
Tensor ste_fake_quantize(const Tensor& x, float scale, int qmin, int qmax);
// LSQ helper (trainable scale)
Tensor lsq_fake_quantize(const Tensor& x, Tensor& scale_param, int qmin, int qmax);

// QAT context for trainer hooks — optionally inserts FakeQuantize nodes
class QATContext {
public:
    explicit QATContext(const QATConfig& cfg) : cfg_(cfg), enabled_(false) {}
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool enabled() const { return enabled_; }
    const QATConfig& config() const { return cfg_; }
    void set_config(const QATConfig& c) { cfg_ = c; }
    // apply fake-quant if enabled else passthrough
    Tensor maybe_quantize(const Tensor& x);
    Tensor maybe_quantize(const Tensor& x, Observer& obs);
    // LSQ variant
    Tensor maybe_lsq_quantize(const Tensor& x, Tensor& scale_param);
private:
    QATConfig cfg_;
    bool enabled_;
};

} // namespace qat
} // namespace quant
