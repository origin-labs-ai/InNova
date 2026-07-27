#pragma once
#include "oil/tensor.h"
#include "oil/types.h"

namespace oil {

// FlashAttention-2: IO-aware tiled attention
// Computes O = softmax(Q*K^T/sqrt(d)) * V using tiling over SRAM
// Memory: O(n) instead of O(n²)
Tensor flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                               const Tensor& mask, float dropout_p = 0.0f, bool causal = true);

struct FlashAttentionConfig {
    int64_t block_size = 64;
    bool causal = true;
    float softmax_scale = 1.0f;
};

class FlashAttention {
public:
    FlashAttention(const FlashAttentionConfig& cfg = FlashAttentionConfig{});
    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                   const Tensor& mask);
private:
    FlashAttentionConfig cfg_;
};

} // namespace oil
