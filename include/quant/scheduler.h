#pragma once
#include <cmath>
#include <vector>
#include <functional>

namespace quant {

class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(int step) = 0;
    virtual float get_last_lr() const { return last_lr_; }
protected:
    float last_lr_ = 0;
};

class ConstantScheduler : public LRScheduler {
public:
    explicit ConstantScheduler(float lr) : lr_(lr) {}
    float get_lr(int) override { last_lr_ = lr_; return lr_; }
private:
    float lr_;
};

class LinearDecayScheduler : public LRScheduler {
public:
    LinearDecayScheduler(float start_lr, float end_lr, int total_steps)
        : start_lr_(start_lr), end_lr_(end_lr), total_steps_(total_steps) {}
    float get_lr(int step) override;
private:
    float start_lr_, end_lr_;
    int total_steps_;
};

class CosineDecayScheduler : public LRScheduler {
public:
    CosineDecayScheduler(float start_lr, float end_lr, int total_steps)
        : start_lr_(start_lr), end_lr_(end_lr), total_steps_(total_steps) {}
    float get_lr(int step) override;
private:
    float start_lr_, end_lr_;
    int total_steps_;
};

class ExponentialDecayScheduler : public LRScheduler {
public:
    ExponentialDecayScheduler(float initial_lr, float decay_rate, int decay_steps)
        : initial_lr_(initial_lr), decay_rate_(decay_rate), decay_steps_(decay_steps) {}
    float get_lr(int step) override;
private:
    float initial_lr_, decay_rate_;
    int decay_steps_;
};

class StepDecayScheduler : public LRScheduler {
public:
    StepDecayScheduler(float initial_lr, int step_size, float gamma = 0.5f)
        : initial_lr_(initial_lr), step_size_(step_size), gamma_(gamma) {}
    float get_lr(int step) override;
private:
    float initial_lr_;
    int step_size_;
    float gamma_;
};

class ReduceLROnPlateauScheduler : public LRScheduler {
public:
    ReduceLROnPlateauScheduler(float initial_lr, float factor = 0.5f,
                               int patience = 5, float min_lr = 1e-6f,
                               float threshold = 1e-4f)
        : initial_lr_(initial_lr), base_lr_(initial_lr), factor_(factor),
          min_lr_(min_lr), threshold_(threshold), patience_(patience) {}
    float get_lr(int step) override;
    void step_metric(float metric);
    void reset() { bad_epochs_ = 0; base_lr_ = initial_lr_; }
private:
    float initial_lr_, base_lr_, factor_, min_lr_, threshold_;
    int patience_, bad_epochs_ = 0, steps_ = 0;
    float best_metric_ = 1e10f;
};

class OneCycleScheduler : public LRScheduler {
public:
    OneCycleScheduler(float max_lr, int total_steps,
                      float pct_start = 0.3f, float div_factor = 25.0f,
                      float final_div_factor = 1e4f)
        : max_lr_(max_lr), total_steps_(total_steps),
          pct_start_(pct_start), div_factor_(div_factor),
          final_div_factor_(final_div_factor) {
        init_lr_ = max_lr_ / div_factor_;
        final_lr_ = init_lr_ / final_div_factor_;
    }
    float get_lr(int step) override;
private:
    float max_lr_, init_lr_, final_lr_;
    int total_steps_;
    float pct_start_, div_factor_, final_div_factor_;
};

class WarmupScheduler : public LRScheduler {
public:
    WarmupScheduler(LRScheduler* wrapped, int warmup_steps, float warmup_start_lr = 0.0f)
        : wrapped_(wrapped), warmup_steps_(warmup_steps), warmup_start_lr_(warmup_start_lr) {}
    float get_lr(int step) override;
    float get_last_lr() const override { return last_lr_; }
private:
    LRScheduler* wrapped_;
    int warmup_steps_;
    float warmup_start_lr_;
};

class SequentialScheduler : public LRScheduler {
public:
    struct Segment { LRScheduler* scheduler; int steps; };
    explicit SequentialScheduler(std::vector<Segment> segments)
        : segments_(std::move(segments)) {}
    float get_lr(int step) override;
private:
    std::vector<Segment> segments_;
};

// Utility: functional scheduler for custom lambda
class LambdaScheduler : public LRScheduler {
public:
    explicit LambdaScheduler(std::function<float(int)> fn) : fn_(std::move(fn)) {}
    float get_lr(int step) override { last_lr_ = fn_(step); return last_lr_; }
private:
    std::function<float(int)> fn_;
};

// ============================================================
// ChainedScheduler — Multiply the output of two schedulers
// ============================================================
class ChainedScheduler : public LRScheduler {
public:
    ChainedScheduler(LRScheduler* primary, LRScheduler* secondary);
    float get_lr(int step) override;
    float get_last_lr() const override;
private:
    LRScheduler* primary_;
    LRScheduler* secondary_;
};

// ============================================================
// CosineWarmupRestart — Cosine annealing with optional warmup
// ============================================================
class CosineWarmupRestart : public LRScheduler {
public:
    CosineWarmupRestart(float base_lr, int total_steps,
                        int warmup_steps = 0, float min_lr = 0.0f);
    float get_lr(int step) override;
private:
    float base_lr_;
    int total_steps_, warmup_steps_;
    float min_lr_;
};

// ============================================================
// PolyScheduler — Polynomial LR decay
// ============================================================
class PolyScheduler : public LRScheduler {
public:
    PolyScheduler(float base_lr, float end_lr, int total_steps,
                  float power = 1.0f);
    float get_lr(int step) override;
private:
    float base_lr_, end_lr_;
    int total_steps_;
    float power_;
};

// ============================================================
// MultiStepDecay — Step decay at specified milestones
// ============================================================
class MultiStepDecay : public LRScheduler {
public:
    MultiStepDecay(float base_lr, std::vector<int> milestones,
                   float gamma = 0.5f);
    float get_lr(int step) override;
private:
    float base_lr_, gamma_;
    std::vector<int> milestones_;
};

// ============================================================
// LinearWarmupDecay — Linear warmup then polynomial decay
// ============================================================
class LinearWarmupDecay : public LRScheduler {
public:
    LinearWarmupDecay(float start_lr, float peak_lr, float end_lr,
                      int warmup_steps, int total_steps,
                      float decay_power = 1.0f);
    float get_lr(int step) override;
private:
    float start_lr_, peak_lr_, end_lr_;
    int warmup_steps_, total_steps_;
    float decay_power_;
};

// ============================================================
// CyclicLR — Triangular cyclic LR with decay
// ============================================================
class CyclicLR : public LRScheduler {
public:
    enum class CycleMode { TRIANGULAR, TRIANGULAR2, EXP_RANGE };
    CyclicLR(float base_lr, float max_lr, int step_size_up,
             int step_size_down = -1, float decay = 0.0f,
             CycleMode mode = CycleMode::TRIANGULAR);
    float get_lr(int step) override;
private:
    float base_lr_, max_lr_, decay_;
    int step_size_up_, step_size_down_, total_size_;
    CycleMode mode_;
};

} // namespace quant