#pragma once
#include "quant/distributed.h"
#include "quant/model.h"
#include "quant/tensor.h"
#include "quant/kv_cache.h"
#include <vector>
#include <memory>
#include <functional>

namespace quant {

struct PipelineStage {
    int device_id;
    int layer_start;
    int layer_end;
    std::unique_ptr<DenseModel> model_segment;
    Tensor input_buffer;
    Tensor output_buffer;
    Tensor grad_buffer;
};

class PipelineParallel {
public:
    PipelineParallel(Model* full_model, int num_stages, int stage_id,
                     int world_size, int world_rank);

    Tensor forward_microbatch(const Tensor& input, const Tensor& positions,
                              KVCache* cache = nullptr);
    void backward_microbatch(const Tensor& global_grad);
    Tensor compute_loss(const Tensor& logits, const Tensor& labels);

    void set_num_microbatches(int m) { num_micro_batches_ = m; }
    int num_micro_batches() const { return num_micro_batches_; }

    Model* model() { return stage_.model_segment.get(); }
    DistributedContext& context() { return ctx_; }

    PipelineStage& stage() { return stage_; }

    Tensor forward(const Tensor& input, const Tensor& positions,
                   const Tensor& mask, KVCache& cache);

    // 1F1B schedule (one forward, one backward) for better memory efficiency
    void forward_1f1b(const Tensor& input, const Tensor& positions,
                      KVCache* cache = nullptr);
    void backward_1f1b();

    // Number of micro-batches currently queued
    int micro_batch_queue_size() const { return (int)micro_batch_outputs_.size(); }

    // Gradient accumulation across micro-batches
    Tensor accumulated_gradient() const;
    void clear_accumulated_gradients();

    // Weights for pipeline balancing (normalized + recorded; used by the
    // profile/balance reporting and available for a dynamic layer
    // redistribution).
    void set_balance_weights(const std::vector<float>& weights);
    const std::vector<float>& balance_weights() const { return balance_weights_; }

    // Get/set model segment reference count to verify model splitting
    int model_param_count() const;

    // Enable/disable profiling
    void set_profiling(bool enabled) { profiling_ = enabled; }
    bool profiling() const { return profiling_; }
    void print_profile() const;

private:
    DistributedContext ctx_;
    int num_stages_;
    int stage_id_;
    int num_micro_batches_ = 4;
    PipelineStage stage_;

    int64_t d_model_;
    int64_t max_seq_len_;

    std::vector<Tensor> micro_batch_outputs_;
    std::vector<Tensor> micro_batch_grads_;
    Tensor accumulated_grad_;

    // Normalized per-stage balance weights (set via set_balance_weights).
    std::vector<float> balance_weights_;

    // The stage's parameters, collected once at construction. backward()
    // registers these with the autograd engine so REAL parameter gradients
    // are computed and accumulated across micro-batches.
    std::vector<Tensor*> stage_params_;

    bool profiling_ = false;
    int64_t profile_forward_ns_ = 0;
    int64_t profile_backward_ns_ = 0;
    int64_t profile_comm_ns_ = 0;

    void split_model(Model* full_model);
    void send_forward(const Tensor& data);
    Tensor recv_forward();
    void send_backward(const Tensor& grad);
    Tensor recv_backward();
};

} // namespace quant
