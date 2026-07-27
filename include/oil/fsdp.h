#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/distributed.h"
#include "oil/optimizer.h"
#include <vector>
#include <string>
#include <memory>

namespace oil {

// ===========================================================================
// FullyShardedDataParallel — ZeRO Stage 1/2/3 parameter sharding
// ===========================================================================
class FullyShardedDataParallel {
public:
    enum class ZeROStage {
        STAGE_1,  // Shard optimizer states
        STAGE_2,  // Shard optimizer states + gradients
        STAGE_3   // Shard optimizer states + gradients + parameters
    };

    struct Config {
        ZeROStage stage = ZeROStage::STAGE_3;
        bool mixed_precision = true;
        float fp16_compute_scale = 1.0f;
        int64_t max_grad_norm = 1;
        bool activation_checkpointing = false;
        bool flatten_parameters = true;
        int64_t memory_limit_bytes = 0;
    };

    struct MemoryProfile {
        int64_t parameters_bytes = 0;
        int64_t gradients_bytes = 0;
        int64_t optimizer_bytes = 0;
        int64_t activations_bytes = 0;
        int64_t communication_bytes = 0;
        int64_t total_bytes = 0;
    };

    FullyShardedDataParallel(Model* model, const Config& config,
                              int world_size, int world_rank);
    ~FullyShardedDataParallel();

    // Forward: AllGather params → compute → discard non-local shards
    Tensor forward(const Tensor& input_ids, const Tensor& positions);

    // Backward: AllGather → compute gradients → ReduceScatter gradients
    void backward(float loss_value);

    // Step: each rank updates only its optimizer shard
    void optimizer_step();

    // Zero gradients
    void zero_grad();

    // Gradient clipping across shards
    void clip_gradients(float max_norm);

    // Mixed precision helpers
    Tensor to_fp16(const Tensor& t) const;
    Tensor to_fp32(const Tensor& t) const;

    // Memory profiling
    MemoryProfile profile_memory() const;
    void print_memory_profile() const;

    // Auto-wrap: wrap TransformerBlock submodules
    static void auto_wrap(DenseModel& model, const Config& config,
                          int world_size, int world_rank);

    // Flat parameter sharding: flatten all params, split evenly
    void flatten_and_shard();
    void unflatten_params();

    // Stage getters
    ZeROStage stage() const { return config_.stage; }
    bool is_mixed_precision() const { return config_.mixed_precision; }
    int world_size() const { return world_size_; }
    int world_rank() const { return world_rank_; }

private:
    Model* model_;
    Config config_;
    int world_size_;
    int world_rank_;
    int step_ = 0;
    DistributedContext ctx_;

    // Per-parameter sharding info
    struct ShardInfo {
        std::string name;
        int64_t total_numel;
        int64_t shard_start;
        int64_t shard_numel;
        Tensor local_param;       // This rank's shard of the parameter
        Tensor local_grad;        // This rank's shard of the gradient
        Tensor fp32_master;       // FP32 master copy for mixed precision
        Tensor optimizer_m;       // AdamW first moment (shard)
        Tensor optimizer_v;       // AdamW second moment (shard)
        Tensor fp16_compute;      // FP16 compute copy
    };

    std::vector<ShardInfo> shards_;
    std::vector<std::string> shard_names_;

    // Flattened parameter state
    Tensor flat_params_;
    Tensor flat_grads_;
    Tensor flat_fp32_master_;
    bool is_flat_ = false;

    // Activation checkpointing
    struct CheckpointEntry {
        Tensor saved_input;
        int64_t layer_idx;
    };
    std::vector<CheckpointEntry> activation_checkpoints_;

    // ZeRO stage implementations
    void init_stage1();
    void init_stage2();
    void init_stage3();

    void forward_stage1(const Tensor& input_ids, const Tensor& positions);
    void forward_stage2(const Tensor& input_ids, const Tensor& positions);
    void forward_stage3(const Tensor& input_ids, const Tensor& positions);

    // AllGather a single shard to get full parameter
    Tensor allgather_param(const ShardInfo& shard);

    // ReduceScatter gradients after backward
    void reduce_scatter_grads();

    // Shard optimizer states
    void init_optimizer_states();

    // Communication
    void allreduce_grad_norm(float& total_norm);
};

// ===========================================================================
// FSDPBlock — wraps a single TransformerBlock with FSDP
// ===========================================================================
class FSDPBlock {
public:
    FSDPBlock() = default;
    FSDPBlock(TransformerBlock* block, int world_size, int world_rank,
              FullyShardedDataParallel::ZeROStage stage);

    // Forward with AllGather before compute, discard after
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx);

    // Backward: re-AllGather, compute grads, ReduceScatter
    void backward(const Tensor& grad_output);

    // Sync gradients after backward
    void sync_gradients();

    // Get memory usage
    int64_t memory_usage() const;

private:
    TransformerBlock* block_;
    int world_size_;
    int world_rank_;
    FullyShardedDataParallel::ZeROStage stage_;

    std::vector<Tensor> local_shards_;
    std::vector<Tensor> local_grads_;
    std::vector<Tensor> fp32_masters_;
};

// ===========================================================================
// ActivationCheckpointing — recompute forward in backward pass
// ===========================================================================
class ActivationCheckpointing {
public:
    ActivationCheckpointing() = default;

    // Save input for a layer (not activations)
    void save_input(int64_t layer_idx, const Tensor& input);

    // Recompute: run layer forward again from saved input
    Tensor recompute(TransformerBlock* block, int64_t layer_idx,
                     const Tensor& positions, const Tensor& mask,
                     KVCache& cache);

    // Clear saved inputs
    void clear();

    // Memory savings report
    int64_t saved_memory() const;

private:
    struct Entry {
        int64_t layer_idx;
        Tensor saved_input;
    };
    std::vector<Entry> entries_;
};

// ===========================================================================
// ZeROOptimizer — per-rank optimizer for sharded parameters
// ===========================================================================
class ZeROOptimizer {
public:
    ZeROOptimizer() = default;
    ZeROOptimizer(float lr, float beta1, float beta2, float eps,
                  float weight_decay, int world_size, int world_rank);

    // Update a single parameter shard
    void step(Tensor& param_shard, const Tensor& grad_shard,
              int64_t step_num);

    // Update with mixed precision (FP32 master weights)
    void step_mixed(Tensor& param_shard, Tensor& fp32_master,
                    const Tensor& grad_shard, int64_t step_num);

    // Get local parameter count
    int64_t param_count() const { return param_count_; }

    // Update total parameter count
    void set_param_count(int64_t n) { param_count_ = n; }

private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int world_size_;
    int world_rank_;
    int64_t param_count_ = 0;
    int64_t step_ = 0;
};

} // namespace oil
