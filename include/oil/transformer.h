#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/kernel.h"
#include "oil/kv_cache.h"
#include <vector>
#include <memory>

namespace oil {

struct TransformerConfig {
    int64_t vocab_size = 32000;
    int64_t hidden_size = 768;
    int64_t num_layers = 12;
    int64_t num_heads = 12;
    int64_t head_dim = 64;
    int64_t ffn_hidden_size = 3072;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int64_t max_seq_len = 2048;
    Activation activation = Activation::SiLU;
    int64_t num_kv_heads = 0;
    bool use_parallel_residual = false;
};

class Embedding {
public:
    Tensor weight;
    Embedding() = default;
    explicit Embedding(int64_t vocab_size, int64_t dim);
    Tensor forward(const Tensor& input_ids) const;
    size_t param_count() const;
};

class Linear {
public:
    Tensor weight;
    Tensor bias;
    Format weight_format = Format::OIL32;
    Linear() = default;
    Linear(int64_t in_features, int64_t out_features);
    Tensor forward(const Tensor& input) const;
    size_t param_count() const;
};

class RMSNorm {
public:
    Tensor weight;
    float eps;
    RMSNorm() : eps(1e-5f) {}
    explicit RMSNorm(int64_t size, float eps = 1e-5f);
    Tensor forward(const Tensor& input) const;
};

class RotaryEmbedding {
public:
    Tensor cos_cached;
    Tensor sin_cached;
    int64_t head_dim;
    float theta;
    RotaryEmbedding() = default;
    RotaryEmbedding(int64_t head_dim, int64_t max_seq_len, float theta = 10000.0f);
    void apply(Tensor& x, int64_t seq_start, int64_t seq_len) const;
};

class Attention {
public:
    Linear q_proj, k_proj, v_proj, o_proj;
    int64_t num_heads, num_kv_heads, head_dim;
    RotaryEmbedding rope;
    Attention() = default;
    explicit Attention(const TransformerConfig& cfg);
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx) const;
};

class FFN {
public:
    Linear gate_proj, up_proj, down_proj;
    Activation activation;
    FFN() : activation(Activation::SiLU) {}
    explicit FFN(const TransformerConfig& cfg);
    FFN(int64_t hidden, int64_t ffn_hidden)
        : gate_proj(hidden, ffn_hidden), up_proj(hidden, ffn_hidden),
          down_proj(ffn_hidden, hidden), activation(Activation::SiLU) {}
    Tensor forward(const Tensor& x) const;
};

class TransformerBlock {
public:
    RMSNorm attention_norm;
    Attention attention;
    RMSNorm ffn_norm;
    FFN ffn;
    bool use_parallel_residual = false;  // GPT-NeoX style: attn+ffn run on same pre-norm input
    TransformerBlock() = default;
    explicit TransformerBlock(const TransformerConfig& cfg);
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx) const;
};

} // namespace oil
