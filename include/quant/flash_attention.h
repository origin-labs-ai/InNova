#pragma once
#include "quant/tensor.h"
#include "quant/types.h"

namespace quant {

// FlashAttention-2: IO-aware tiled attention
// Computes O = softmax(Q*K^T/sqrt(d)) * V using tiling over SRAM
// Memory: O(n) instead of O(n²)
Tensor flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                               const Tensor& mask, float dropout_p = 0.0f, bool causal = true);

struct FlashAttentionConfig {
    int64_t block_size = 64;
    bool causal = true;
    float softmax_scale = 1.0f;
    float dropout = 0.0f;   // attention dropout probability (0 disables)
};

class FlashAttention {
public:
    FlashAttention(const FlashAttentionConfig& cfg = FlashAttentionConfig{});
    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                   const Tensor& mask);
private:
    FlashAttentionConfig cfg_;
};

} // namespace quant
