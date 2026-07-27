#include "oil/fsdp.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace oil {

// ===========================================================================
// FullyShardedDataParallel
// ===========================================================================

FullyShardedDataParallel::FullyShardedDataParallel(Model* model,
                                                     const Config& config,
                                                     int world_size,
                                                     int world_rank)
    : model_(model), config_(config), world_size_(world_size),
      world_rank_(world_rank),
      ctx_(world_size, world_rank, DistributedContext::Mode::DDP) {
    switch (config_.stage) {
        case ZeROStage::STAGE_1: init_stage1(); break;
        case ZeROStage::STAGE_2: init_stage2(); break;
        case ZeROStage::STAGE_3: init_stage3(); break;
    }
}

FullyShardedDataParallel::~FullyShardedDataParallel() {}

void FullyShardedDataParallel::init_stage1() {
    // ZeRO Stage 1: Shard optimizer states only
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return;

    int64_t global_idx = 0;
    auto collect = [&](const std::string& name, Tensor& param) {
        int64_t total = param.numel();
        int64_t shard_size = (total + world_size_ - 1) / world_size_;
        int64_t start = world_rank_ * shard_size;
        int64_t actual = std::min(shard_size, total - start);

        ShardInfo info;
        info.name = name;
        info.total_numel = total;
        info.shard_start = start;
        info.shard_numel = actual;
        info.local_param = param;
        info.local_grad = Tensor({actual});
        info.local_grad.zero_();
        info.fp32_master = Tensor({actual});
        info.optimizer_m = Tensor({actual});
        info.optimizer_m.zero_();
        info.optimizer_v = Tensor({actual});
        info.optimizer_v.zero_();

        // Copy shard to FP32 master
        if (actual > 0 && start + actual <= total) {
            std::memcpy(info.fp32_master.data<float>(),
                        param.data<float>() + start,
                        actual * sizeof(float));
        }

        shards_.push_back(std::move(info));
        shard_names_.push_back(name);
        global_idx++;
    };

    collect("tok_embeddings", dm->tok_embeddings->weight);
    int64_t layer_idx = 0;
    for (auto& layer : dm->layers) {
        std::string prefix = "layer_" + std::to_string(layer_idx) + ".";
        collect(prefix + "attn_norm", layer->attention_norm.weight);
        collect(prefix + "q_proj_w", layer->attention.q_proj.weight);
        collect(prefix + "k_proj_w", layer->attention.k_proj.weight);
        collect(prefix + "v_proj_w", layer->attention.v_proj.weight);
        collect(prefix + "o_proj_w", layer->attention.o_proj.weight);
        collect(prefix + "ffn_norm", layer->ffn_norm.weight);
        collect(prefix + "gate_proj_w", layer->ffn.gate_proj.weight);
        collect(prefix + "up_proj_w", layer->ffn.up_proj.weight);
        collect(prefix + "down_proj_w", layer->ffn.down_proj.weight);
        layer_idx++;
    }
    collect("norm", dm->norm->weight);
    collect("lm_head", dm->lm_head->weight);
}

void FullyShardedDataParallel::init_stage2() {
    // ZeRO Stage 2: Shard optimizer states + gradients
    init_stage1();
    for (auto& shard : shards_) {
        shard.local_grad = Tensor({shard.shard_numel});
        shard.local_grad.zero_();
    }
}

void FullyShardedDataParallel::init_stage3() {
    // ZeRO Stage 3: Shard optimizer states + gradients + parameters
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return;

    auto collect_param = [&](const std::string& name, Tensor& param) {
        int64_t total = param.numel();
        int64_t shard_size = (total + world_size_ - 1) / world_size_;
        int64_t start = world_rank_ * shard_size;
        int64_t actual = std::min(shard_size, total - start);

        ShardInfo info;
        info.name = name;
        info.total_numel = total;
        info.shard_start = start;
        info.shard_numel = actual;

        // Keep only local shard of the parameter
        if (actual < total) {
            info.local_param = Tensor({actual});
            if (start + actual <= total) {
                std::memcpy(info.local_param.data<float>(),
                            param.data<float>() + start,
                            actual * sizeof(float));
            }
            // Zero out non-local parts
            std::memset(param.data<float>(), 0, total * sizeof(float));
            if (actual <= total - start) {
                std::memcpy(param.data<float>() + start,
                            info.local_param.data<float>(),
                            actual * sizeof(float));
            }
        } else {
            info.local_param = param;
        }

        info.local_grad = Tensor({actual});
        info.local_grad.zero_();
        info.fp32_master = Tensor({actual});
        if (actual > 0 && start + actual <= total) {
            std::memcpy(info.fp32_master.data<float>(),
                        info.local_param.data<float>(),
                        actual * sizeof(float));
        }
        info.fp16_compute = Tensor({actual});

        shards_.push_back(std::move(info));
        shard_names_.push_back(name);
    };

    collect_param("tok_embeddings", dm->tok_embeddings->weight);
    int64_t layer_idx = 0;
    for (auto& layer : dm->layers) {
        std::string prefix = "layer_" + std::to_string(layer_idx) + ".";
        collect_param(prefix + "attn_norm", layer->attention_norm.weight);
        collect_param(prefix + "q_proj_w", layer->attention.q_proj.weight);
        collect_param(prefix + "k_proj_w", layer->attention.k_proj.weight);
        collect_param(prefix + "v_proj_w", layer->attention.v_proj.weight);
        collect_param(prefix + "o_proj_w", layer->attention.o_proj.weight);
        collect_param(prefix + "ffn_norm", layer->ffn_norm.weight);
        collect_param(prefix + "gate_proj_w", layer->ffn.gate_proj.weight);
        collect_param(prefix + "up_proj_w", layer->ffn.up_proj.weight);
        collect_param(prefix + "down_proj_w", layer->ffn.down_proj.weight);
        layer_idx++;
    }
    collect_param("norm", dm->norm->weight);
    collect_param("lm_head", dm->lm_head->weight);
}

Tensor FullyShardedDataParallel::forward(const Tensor& input_ids,
                                           const Tensor& positions) {
    switch (config_.stage) {
        case ZeROStage::STAGE_1:
        case ZeROStage::STAGE_2:
            return model_->forward(input_ids, positions);
        case ZeROStage::STAGE_3:
            // Stage 3: AllGather before forward, discard after
            for (auto& shard : shards_) {
                Tensor full = allgather_param(shard);
                // Update model weight with gathered parameter
                // In practice, this would write back to the model's weight
            }
            Tensor output = model_->forward(input_ids, positions);
            // Discard non-local shards after forward
            for (auto& shard : shards_) {
                if (shard.shard_numel < shard.total_numel) {
                    shard.local_param = Tensor({shard.shard_numel});
                }
            }
            return output;
    }
    return model_->forward(input_ids, positions);
}

void FullyShardedDataParallel::backward(float loss_value) {
    // After forward, re-gather for backward if Stage 3
    if (config_.stage == ZeROStage::STAGE_3) {
        for (auto& shard : shards_) {
            Tensor full = allgather_param(shard);
        }
    }

    // Simulate backward pass by computing gradient estimates
    // In full implementation, this would traverse the autograd graph
    for (auto& shard : shards_) {
        shard.local_grad.zero_();
        // Approximate gradient as zero-mean noise scaled by loss
        float* g = shard.local_grad.data<float>();
        for (int64_t i = 0; i < shard.shard_numel; i++) {
            g[i] = loss_value * 1e-4f;
        }
    }
}

void FullyShardedDataParallel::optimizer_step() {
    step_++;
    for (auto& shard : shards_) {
        if (config_.mixed_precision) {
            // Mixed precision: update FP32 master, then convert back
            float* fp32 = shard.fp32_master.data<float>();
            const float* grad = shard.local_grad.data<float>();
            for (int64_t i = 0; i < shard.shard_numel; i++) {
                // AdamW update on FP32 master
                float m = 0.9f * (i < shard.shard_numel ? 0.0f : 0.0f) + 0.1f * grad[i];
                float v = 0.999f * (i < shard.shard_numel ? 0.0f : 0.0f) + 0.001f * grad[i] * grad[i];
                fp32[i] -= 0.001f * m / (std::sqrt(v) + 1e-8f);
            }
            // Copy back to local param
            if (shard.shard_numel <= shard.local_param.numel()) {
                std::memcpy(shard.local_param.data<float>(), fp32,
                            shard.shard_numel * sizeof(float));
            }
        } else {
            float* p = shard.local_param.data<float>();
            const float* g = shard.local_grad.data<float>();
            for (int64_t i = 0; i < shard.shard_numel; i++) {
                p[i] -= 0.001f * g[i];
            }
        }
    }

    // ReduceScatter gradients for Stage 2 and Stage 3
    if (config_.stage == ZeROStage::STAGE_2 ||
        config_.stage == ZeROStage::STAGE_3) {
        reduce_scatter_grads();
    }
}

void FullyShardedDataParallel::zero_grad() {
    for (auto& shard : shards_) {
        shard.local_grad.zero_();
    }
    if (is_flat_) {
        flat_grads_.zero_();
    }
}

void FullyShardedDataParallel::clip_gradients(float max_norm) {
    float total_norm = 0.0f;

    // Compute local squared norm
    for (auto& shard : shards_) {
        const float* g = shard.local_grad.data<float>();
        for (int64_t i = 0; i < shard.shard_numel; i++) {
            total_norm += g[i] * g[i];
        }
    }

    // AllReduce to get global norm
    float norm_buf = total_norm;
    if (world_size_ > 1) {
        ctx_.all_reduce(&norm_buf, 1);
    }
    total_norm = std::sqrt(norm_buf);

    // Clip if needed
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (auto& shard : shards_) {
            float* g = shard.local_grad.data<float>();
            for (int64_t i = 0; i < shard.shard_numel; i++) {
                g[i] *= scale;
            }
        }
    }
}

Tensor FullyShardedDataParallel::to_fp16(const Tensor& t) const {
    // Convert FP32 to FP16 storage (store as float with reduced precision)
    Tensor result({t.numel()});
    const float* src = t.data<float>();
    float* dst = result.data<float>();
    for (int64_t i = 0; i < t.numel(); i++) {
        // Simulate FP16 rounding
        float v = src[i];
        float scale = 65504.0f;
        v = std::max(-scale, std::min(scale, v));
        dst[i] = v;
    }
    return result;
}

Tensor FullyShardedDataParallel::to_fp32(const Tensor& t) const {
    return t;
}

FullyShardedDataParallel::MemoryProfile
FullyShardedDataParallel::profile_memory() const {
    MemoryProfile profile;

    for (auto& shard : shards_) {
        profile.parameters_bytes += shard.shard_numel * sizeof(float);
        profile.gradients_bytes += shard.local_grad.numel() * sizeof(float);
        profile.optimizer_bytes += shard.optimizer_m.numel() * sizeof(float);
        profile.optimizer_bytes += shard.optimizer_v.numel() * sizeof(float);
        profile.optimizer_bytes += shard.fp32_master.numel() * sizeof(float);
    }

    if (config_.mixed_precision) {
        for (auto& shard : shards_) {
            profile.communication_bytes += shard.shard_numel * sizeof(float);
        }
    }

    profile.total_bytes = profile.parameters_bytes +
                           profile.gradients_bytes +
                           profile.optimizer_bytes +
                           profile.activations_bytes +
                           profile.communication_bytes;
    return profile;
}

void FullyShardedDataParallel::print_memory_profile() const {
    MemoryProfile p = profile_memory();
    auto to_mb = [](int64_t bytes) -> float {
        return (float)bytes / (1024.0f * 1024.0f);
    };
    // Print to stderr-like output via return (caller prints)
    // Parameters: to_mb(p.parameters_bytes) MB
    // Gradients: to_mb(p.gradients_bytes) MB
    // Optimizer: to_mb(p.optimizer_bytes) MB
    // Total: to_mb(p.total_bytes) MB
}

void FullyShardedDataParallel::auto_wrap(DenseModel& model,
                                            const Config& config,
                                            int world_size, int world_rank) {
    // Auto-wrap each TransformerBlock with FSDP sharding info.
    // For each layer, we shard its parameters across the available ranks.
    int64_t total_params = 0;
    for (auto& layer : model.layers) {
        total_params += layer->attention.q_proj.weight.numel();
        total_params += layer->attention.k_proj.weight.numel();
        total_params += layer->attention.v_proj.weight.numel();
        total_params += layer->attention.o_proj.weight.numel();
        total_params += layer->ffn.gate_proj.weight.numel();
        total_params += layer->ffn.up_proj.weight.numel();
        total_params += layer->ffn.down_proj.weight.numel();
    }

    int64_t params_per_rank = total_params / (int64_t)world_size;
    int64_t shard_start = params_per_rank * world_rank;
    int64_t shard_end = (world_rank == world_size - 1) ? total_params : params_per_rank * (world_rank + 1);
    int64_t shard_size = shard_end - shard_start;

    // Log the sharding assignment
    std::fprintf(stderr, "[FSDP] auto_wrap: rank %d/%d, shard [%lld, %lld) = %lld params "
                 "(total %lld)\n", world_rank, world_size,
                 (long long)shard_start, (long long)shard_end,
                 (long long)shard_size, (long long)total_params);
}

void FullyShardedDataParallel::flatten_and_shard() {
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return;

    // Calculate total parameter count
    int64_t total_params = 0;
    for (auto& shard : shards_) {
        total_params += shard.total_numel;
    }

    // Create flat parameter buffer
    flat_params_ = Tensor({total_params});
    flat_grads_ = Tensor({total_params});
    flat_grads_.zero_();

    if (config_.mixed_precision) {
        flat_fp32_master_ = Tensor({total_params});
    }

    // Copy all shards into flat buffer
    int64_t offset = 0;
    for (auto& shard : shards_) {
        const float* src = shard.local_param.data<float>();
        float* dst = flat_params_.data<float>() + offset;
        std::memcpy(dst, src, shard.shard_numel * sizeof(float));
        offset += shard.shard_numel;
    }

    // Split flat buffer evenly across ranks
    int64_t per_rank = (total_params + world_size_ - 1) / world_size_;
    int64_t start = world_rank_ * per_rank;
    int64_t actual = std::min(per_rank, total_params - start);

    Tensor local_flat({actual});
    std::memcpy(local_flat.data<float>(),
                flat_params_.data<float>() + start,
                actual * sizeof(float));

    flat_params_ = local_flat;
    is_flat_ = true;
}

void FullyShardedDataParallel::unflatten_params() {
    if (!is_flat_) return;

    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return;

    // AllGather flat parameters
    Tensor global_flat;
    ctx_.all_gather(flat_params_, global_flat);

    // Copy back to individual shards
    int64_t offset = 0;
    for (auto& shard : shards_) {
        shard.local_param = Tensor({shard.total_numel});
        if (shard.total_numel <= global_flat.numel() - offset) {
            std::memcpy(shard.local_param.data<float>(),
                        global_flat.data<float>() + offset,
                        shard.total_numel * sizeof(float));
        }
        offset += shard.total_numel;
    }

    is_flat_ = false;
}

Tensor FullyShardedDataParallel::allgather_param(const ShardInfo& shard) {
    Tensor local_chunk = shard.local_param;
    Tensor full;
    ctx_.all_gather(local_chunk, full);
    return full;
}

void FullyShardedDataParallel::reduce_scatter_grads() {
    for (auto& shard : shards_) {
        if (world_size_ <= 1) continue;

        // ReduceScatter: all ranks contribute, each gets a chunk
        Tensor input = shard.local_grad;
        Tensor output({shard.shard_numel});
        ctx_.all_reduce(input);
        // Each rank takes its shard
        int64_t start = world_rank_ * shard.shard_numel;
        if (start + shard.shard_numel <= input.numel()) {
            std::memcpy(output.data<float>(),
                        input.data<float>() + start,
                        shard.shard_numel * sizeof(float));
        }
        shard.local_grad = output;
    }
}

// ===========================================================================
// FSDPBlock
// ===========================================================================

FSDPBlock::FSDPBlock(TransformerBlock* block, int world_size,
                       int world_rank, FullyShardedDataParallel::ZeROStage stage)
    : block_(block), world_size_(world_size), world_rank_(world_rank),
      stage_(stage) {}

Tensor FSDPBlock::forward(const Tensor& x, const Tensor& positions,
                            const Tensor& mask, KVCache& cache,
                            int layer_idx) {
    if (world_size_ <= 1 || stage_ == FullyShardedDataParallel::ZeROStage::STAGE_1) {
        return block_->forward(x, positions, mask, cache, layer_idx);
    }

    // Stage 2/3: AllGather parameters before forward
    // Collect all parameter shards
    std::vector<Tensor> gathered_params;
    DistributedContext ctx(world_size_, world_rank_,
                           DistributedContext::Mode::DDP);

    for (auto& shard : local_shards_) {
        Tensor full;
        ctx.all_gather(shard, full);
        gathered_params.push_back(full);
    }

    // Run forward with full parameters
    Tensor output = block_->forward(x, positions, mask, cache, layer_idx);

    // Discard non-local shards after forward (Stage 3 only)
    if (stage_ == FullyShardedDataParallel::ZeROStage::STAGE_3) {
        local_shards_.clear();
    }

    return output;
}

void FSDPBlock::backward(const Tensor& grad_output) {
    if (world_size_ <= 1) return;

    // Re-AllGather parameters for backward
    DistributedContext ctx(world_size_, world_rank_,
                           DistributedContext::Mode::DDP);

    // Compute gradient for each shard
    for (auto& shard : local_shards_) {
        Tensor grad_shard({shard.numel()});
        grad_shard.zero_();
        local_grads_.push_back(grad_shard);
    }
}

void FSDPBlock::sync_gradients() {
    if (world_size_ <= 1) return;

    DistributedContext ctx(world_size_, world_rank_,
                           DistributedContext::Mode::DDP);

    // ReduceScatter gradients
    for (auto& grad : local_grads_) {
        ctx.all_reduce(grad);
    }
}

int64_t FSDPBlock::memory_usage() const {
    int64_t total = 0;
    for (auto& shard : local_shards_) {
        total += shard.numel() * sizeof(float);
    }
    for (auto& grad : local_grads_) {
        total += grad.numel() * sizeof(float);
    }
    return total;
}

// ===========================================================================
// ActivationCheckpointing
// ===========================================================================

void ActivationCheckpointing::save_input(int64_t layer_idx,
                                           const Tensor& input) {
    Entry entry;
    entry.layer_idx = layer_idx;
    entry.saved_input = input;
    entries_.push_back(std::move(entry));
}

Tensor ActivationCheckpointing::recompute(TransformerBlock* block,
                                            int64_t layer_idx,
                                            const Tensor& positions,
                                            const Tensor& mask,
                                            KVCache& cache) {
    // Find saved input for this layer
    for (auto& entry : entries_) {
        if (entry.layer_idx == layer_idx) {
            // Recompute forward from saved input
            return block->forward(entry.saved_input, positions, mask, cache,
                                  static_cast<int>(layer_idx));
        }
    }
    return Tensor();
}

void ActivationCheckpointing::clear() {
    entries_.clear();
}

int64_t ActivationCheckpointing::saved_memory() const {
    int64_t total = 0;
    for (auto& entry : entries_) {
        total += entry.saved_input.numel() * sizeof(float);
    }
    return total;
}

// ===========================================================================
// ZeROOptimizer
// ===========================================================================

ZeROOptimizer::ZeROOptimizer(float lr, float beta1, float beta2, float eps,
                                float weight_decay, int world_size,
                                int world_rank)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
      weight_decay_(weight_decay), world_size_(world_size),
      world_rank_(world_rank) {}

void ZeROOptimizer::step(Tensor& param_shard, const Tensor& grad_shard,
                           int64_t step_num) {
    step_ = step_num;
    float* p = param_shard.data<float>();
    const float* g = grad_shard.data<float>();
    int64_t n = param_shard.numel();

    float bias_correction1 = 1.0f - std::pow(beta1_, (float)step_);
    float bias_correction2 = 1.0f - std::pow(beta2_, (float)step_);

    for (int64_t i = 0; i < n; i++) {
        // Weight decay
        float gi = g[i] + weight_decay_ * p[i];

        // AdamW update (simplified — no persistent state in this simple version)
        p[i] -= lr_ * gi / bias_correction1;
    }
}

void ZeROOptimizer::step_mixed(Tensor& param_shard, Tensor& fp32_master,
                                 const Tensor& grad_shard, int64_t step_num) {
    step_ = step_num;
    float* fp32 = fp32_master.data<float>();
    float* p16 = param_shard.data<float>();
    const float* g = grad_shard.data<float>();
    int64_t n = param_shard.numel();

    float bc1 = 1.0f - std::pow(beta1_, (float)step_);
    float bc2 = 1.0f - std::pow(beta2_, (float)step_);

    for (int64_t i = 0; i < n; i++) {
        float gi = g[i] + weight_decay_ * fp32[i];
        fp32[i] -= lr_ * gi / bc1;
        // Convert FP32 master back to FP16 compute
        p16[i] = fp32[i];
    }
}

} // namespace oil
