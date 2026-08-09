#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/distributed.h"
#include "quant/optimizer.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace quant {

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
        Tensor* model_param = nullptr;   // The model's actual parameter tensor
        bool local_param_is_full = false;// local_param aliases the FULL model param
        Tensor local_param;       // This rank's shard of the parameter
        Tensor local_grad;        // This rank's shard of the gradient
        Tensor fp32_master;       // FP32 master copy for mixed precision
        Tensor optimizer_m;       // legacy optimizer state (unused after Adafactor switch)
        Tensor optimizer_v;       // legacy optimizer state (unused after Adafactor switch)
        Tensor fp16_compute;      // FP16 compute copy
    };

    std::vector<ShardInfo> shards_;
    std::vector<std::string> shard_names_;

    // Adafactor drives every rank's shard update. shard_opt_params_ holds the
    // 2-D views registered with it (each aliases a shard's FP32 master copy).
    Adafactor shard_opt_;
    std::vector<Tensor> shard_opt_params_;
    void register_shard_params();

    // The forward output saved for the real autograd backward pass.
    Tensor last_output_;

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

    // Parameter tensors of the wrapped block (enumerated once) and the
    // forward output saved for backward. local_shards_[i] is this rank's
    // contiguous slice of parameter i; local_grads_[i] is this rank's slice
    // of parameter i's gradient (computed by the autograd engine).
    std::vector<Tensor*> param_ptrs_;
    Tensor last_output_;
    std::vector<Tensor> local_shards_;
    std::vector<Tensor> local_grads_;
    std::vector<Tensor> fp32_masters_;

    void enumerate_params();
    void ensure_local_shards();
    void gather_and_install();
    void shard_gradients();
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

    // Update a single parameter shard (real Adafactor with persistent
    // factorized state per shard buffer)
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

    // Per-parameter Adafactor state, keyed by the shard's data buffer so a
    // repeated step() on the same shard keeps its row/column factors. Each
    // ParamOpt owns a 2-D view of the shard buffer (Adafactor factorizes over
    // two dims) plus the Adafactor that updates just that shard.
    struct ParamOpt {
        Tensor view;
        Adafactor adafactor;
    };
    std::unordered_map<const float*, ParamOpt> optims_;

    ParamOpt& opt_for(Tensor& buffer);
};

} // namespace quant
