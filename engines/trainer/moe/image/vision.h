#pragma once
#include "quant/tensor.h"
#include "quant/transformer.h"
#include <vector>

namespace quant {
namespace multimodal {

class ViTEncoder {
public:
    ViTEncoder(int64_t img_size, int64_t patch_size, int64_t hidden_size,
               int64_t num_layers, int64_t num_heads);
    Tensor forward(const Tensor& images);

    Tensor patch_embed;
    Tensor pos_embed;
    Tensor cls_token;
    std::vector<TransformerBlock> blocks;
    int64_t hidden_size;
    int64_t num_patches;
};

} // namespace multimodal
} // namespace quant
