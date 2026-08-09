#pragma once
#include "quant/tensor.h"

namespace quant {
namespace multimodal {

class TextEncoder {
public:
    TextEncoder(int64_t vocab_size, int64_t hidden_size, int64_t max_len);
    Tensor forward(const Tensor& input_ids);
    Tensor embed;
    Tensor pos_embed;
    int64_t hidden_size;
    int64_t max_len;
};

} // namespace multimodal
} // namespace quant
