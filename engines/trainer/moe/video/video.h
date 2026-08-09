#pragma once
#include "quant/tensor.h"
#include "quant/transformer.h"
#include <vector>

namespace quant {
namespace multimodal {

class VideoEncoder {
public:
    VideoEncoder(int64_t tube_size, int64_t img_size, int64_t patch_size,
                 int64_t hidden_size, int64_t num_layers, int64_t num_heads);
    Tensor forward(const Tensor& video);

    Tensor conv3d_proj;
    Tensor pos_embed;
    Tensor cls_token;
    std::vector<TransformerBlock> blocks;
    int64_t hidden_size;
    int64_t num_tubes;
    int64_t patch_size;
    int64_t tube_size;
};

} // namespace multimodal
} // namespace quant
