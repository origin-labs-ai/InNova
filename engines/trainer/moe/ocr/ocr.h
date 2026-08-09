#pragma once
#include "quant/tensor.h"
#include "quant/transformer.h"
#include <vector>

namespace quant {
namespace multimodal {

class OCREncoder {
public:
    OCREncoder(int64_t hidden_size, int64_t num_layers, int64_t num_heads);
    Tensor forward(const Tensor& image);

    Tensor conv_proj;
    Tensor pos_embed;
    std::vector<TransformerBlock> blocks;
    int64_t hidden_size;
};

} // namespace multimodal
} // namespace quant
