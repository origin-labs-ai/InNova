#pragma once
#include "oil/distributed.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <functional>

namespace oil {

enum class GradCompression { NONE, TOP_K_SPARSE, FP8_QUANT };
enum class DDPMode { SYNC, ASYNC };

struct Bucket {
    std::vector<Tensor*> params;
    Tensor buffer;
    size_t size = 0;
    int64_t numel = 0;
};

class DistributedDataParallel {
public:
    DistributedDataParallel(Model* model, int world_size, int world_rank,
                            DDPMode mode = DDPMode::SYNC);
    ~DistributedDataParallel();

    Tensor forward(const Tensor& input, const Tensor& positions);
    void sync_gradients();
    void sync_buffers(const std::vector<Tensor*>& buffers);

    void set_grad_compression(GradCompression comp, float sparse_ratio = 0.01f);
    void set_bucket_size(int64_t bucket_numel = 1048576);

    Model* model() { return model_; }
    DistributedContext& context() { return ctx_; }

    int world_size() const { return ctx_.world_size(); }
    int rank() const { return ctx_.world_rank(); }

    // Compute global gradient norm across all parameters
    float global_grad_norm();

    // Apply gradient clipping at the DDP level
    void clip_gradients(float max_norm);

    // Average model parameters across all ranks (post-optimizer)
    void average_model(float weight = 1.0f);

    // Synchronize all model parameters from rank 0
    void broadcast_model(int src_rank = 0);

    // Gradient accumulation over N micro-batches
    void set_grad_accumulation_steps(int steps) { grad_accum_steps_ = steps; }
    int grad_accumulation_steps() const { return grad_accum_steps_; }
    void zero_grad();
    int grad_accum_count() const { return grad_accum_count_; }
    void reset_grad_accum_count() { grad_accum_count_ = 0; }

    // Gradient noise injection
    void inject_gradient_noise(int step, float eta, float gamma);

private:
    DistributedContext ctx_;
    Model* model_;
    DDPMode mode_;
    GradCompression compression_ = GradCompression::NONE;
    float sparse_ratio_ = 0.01f;
    int64_t bucket_numel_ = 1048576;

    std::vector<Bucket> buckets_;
    std::vector<Tensor*> all_params_;

    void collect_params();
    void build_buckets();
    void allreduce_bucket(Bucket& bucket);
    void compress_topk_sparse(Tensor& grad, Tensor& values, Tensor& indices, int64_t k);
    void decompress_topk_sparse(Tensor& grad, const Tensor& values, const Tensor& indices, int64_t n);
    void compress_fp8(const Tensor& src, std::vector<uint8_t>& dst, float& scale);
    void decompress_fp8(std::vector<uint8_t>& src, float scale, Tensor& dst);

    int grad_accum_steps_ = 1;
    int grad_accum_count_ = 0;
    std::vector<Tensor> grad_accum_buffers_;
    void init_grad_accum_buffers();

    std::mutex async_mutex_;
    std::thread async_thread_;
    std::atomic<bool> async_running_{false};
    void async_sync_loop();
};

} // namespace oil
