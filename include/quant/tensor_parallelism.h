#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/distributed.h"
#include <vector>
#include <string>
#include <memory>

namespace quant {

// ===========================================================================
// TensorParallelManager — splits weight matrices across N ranks
// ===========================================================================
class TensorParallelManager {
public:
    TensorParallelManager(Model* model, int world_size, int world_rank);
    ~TensorParallelManager();

    int world_size() const { return world_size_; }
    int world_rank() const { return world_rank_; }

    // Column-parallel: split weight matrix columns across ranks
    // Used for output projections (q_proj, k_proj, v_proj, gate_proj, up_proj)
    void column_parallel_split(const Tensor& weight, Tensor& local_weight);
    void column_parallel_split(const Tensor& weight, Tensor& local_weight,
                               int64_t rank, int64_t size);

    // Row-parallel: split weight matrix rows across ranks
    // Used for input projections (o_proj, down_proj)
    void row_parallel_split(const Tensor& weight, Tensor& local_weight);
    void row_parallel_split(const Tensor& weight, Tensor& local_weight,
                            int64_t rank, int64_t size);

    // Column-parallel linear: Y_local = X @ W_local
    // Each rank computes partial output, then AllReduce for full result
    Tensor column_parallel_linear(const Tensor& input, const Tensor& local_weight);

    // Row-parallel linear: Y = AllReduce(X_local @ W_local)
    // Each rank has a shard of rows, compute partial, then AllReduce
    Tensor row_parallel_linear(const Tensor& input, const Tensor& local_weight);

    // Attention head partitioning: each rank gets a subset of heads
    void partition_heads(int64_t total_heads, int64_t& heads_per_rank,
                         int64_t& head_start);

    // Sequence parallel: split sequence dimension for LayerNorm/Dropout
    Tensor sequence_parallel_scatter(const Tensor& x, int64_t seq_dim = 1);
    Tensor sequence_parallel_gather(const Tensor& x, int64_t seq_dim = 1);

    // Communication primitives
    void allreduce_tensor(Tensor& tensor);
    void allgather_tensor(const Tensor& local, Tensor& global);
    void reduce_scatter_tensor(const Tensor& input, Tensor& output);

    // Partition strategy: auto-select based on model size
    enum class PartitionStrategy {
        COLUMN_ONLY,
        ROW_ONLY,
        COLUMN_THEN_ROW,
        AUTO
    };
    PartitionStrategy select_strategy(const TransformerConfig& cfg) const;

    // Apply tensor parallelism to a full TransformerBlock
    void parallelize_block(TransformerBlock& block);
    void parallelize_model(DenseModel& model);

    // Reconstruct full weight from local shards (for save/export)
    Tensor reconstruct_weight(const Tensor& local_weight, int64_t full_dim,
                              bool is_column);

    // Communication cost estimation
    int64_t communication_volume(const TransformerConfig& cfg) const;

private:
    Model* model_;
    int world_size_;
    int world_rank_;
    DistributedContext ctx_;
    PartitionStrategy strategy_;

    void apply_column_parallel_attention(Attention& attn);
    void apply_row_parallel_attention(Attention& attn);
    void apply_column_parallel_ffn(FFN& ffn);
    void apply_row_parallel_ffn(FFN& ffn);
};

// ===========================================================================
// ColumnParallelLinear — holds local shard, AllReduce after matmul
// ===========================================================================
class ColumnParallelLinear {
public:
    ColumnParallelLinear() = default;
    ColumnParallelLinear(int64_t in_features, int64_t out_features,
                         int world_size, int world_rank);

    Tensor forward(const Tensor& input);
    Tensor& local_weight() { return local_weight_; }
    Tensor& local_bias() { return local_bias_; }
    int64_t local_out_features() const;

private:
    Tensor local_weight_;
    Tensor local_bias_;
    int64_t in_features_;
    int64_t out_features_;
    int world_size_;
    int world_rank_;
    bool has_bias_;
};

// ===========================================================================
// RowParallelLinear — holds local shard, AllReduce after matmul
// ===========================================================================
class RowParallelLinear {
public:
    RowParallelLinear() = default;
    RowParallelLinear(int64_t in_features, int64_t out_features,
                      int world_size, int world_rank);

    Tensor forward(const Tensor& input);
    Tensor& local_weight() { return local_weight_; }
    Tensor& local_bias() { return local_bias_; }
    int64_t local_in_features() const;

private:
    Tensor local_weight_;
    Tensor local_bias_;
    int64_t in_features_;
    int64_t out_features_;
    int world_size_;
    int world_rank_;
    bool has_bias_;
};

// ===========================================================================
// HeadPartitioner — partitions attention heads across ranks
// ===========================================================================
class HeadPartitioner {
public:
    HeadPartitioner() = default;
    HeadPartitioner(int64_t num_heads, int64_t head_dim,
                    int world_size, int world_rank);

    // Get this rank's head range
    int64_t head_start() const { return head_start_; }
    int64_t heads_per_rank() const { return heads_per_rank_; }
    int64_t head_end() const { return head_start_ + heads_per_rank_; }

    // Partition Q, K, V tensors along head dimension
    Tensor partition_heads(const Tensor& qkv, int64_t seq_len) const;

    // Merge partial attention outputs from all heads on this rank
    Tensor merge_heads(const Tensor& attn_out, int64_t seq_len) const;

    // AllReduce across ranks after output projection
    void sync_output(Tensor& output) const;

private:
    int64_t num_heads_;
    int64_t head_dim_;
    int64_t heads_per_rank_;
    int64_t head_start_;
    int world_size_;
    int world_rank_;
};

// ===========================================================================
// SequenceParallel — splits/gathers sequence dimension
// ===========================================================================
class SequenceParallel {
public:
    SequenceParallel() = default;
    SequenceParallel(int world_size, int world_rank);

    // Scatter sequence across ranks: input [B, S, D] -> [B, S/N, D] per rank
    Tensor scatter(const Tensor& x, int64_t seq_dim = 1) const;

    // Gather sequence from all ranks: [B, S/N, D] -> [B, S, D]
    Tensor gather(const Tensor& x, int64_t seq_dim = 1) const;

    // AllReduce after sequence-parallel matmul
    void allreduce(Tensor& x) const;

    int world_size() const { return world_size_; }
    int world_rank() const { return world_rank_; }

private:
    int world_size_;
    int world_rank_;
};

} // namespace quant
