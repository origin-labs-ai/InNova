#include "codec.h"
#include "oil/codebook.h"
#include "oil/tensor.h"

#include <cstring>
#include <algorithm>
#include <vector>

namespace oil {
namespace oil8 {

OIL8Encoder::OIL8Encoder(const CodecConfig& cfg)
    : config_(cfg) {}

CodecConfig OIL8Encoder::config() const {
    return config_;
}

EncodedBlock OIL8Encoder::encode(const float* weights, uint32_t num_weights, uint32_t block_id) {
    EncodedBlock block;
    block.block_id = block_id;
    block.num_weights = num_weights;
    block.indices.resize(num_weights);

    block.codebook.train(weights, num_weights);

    for (uint32_t i = 0; i < num_weights; i++) {
        block.indices[i] = block.codebook.quantize(weights[i]);
    }

    return block;
}

std::vector<EncodedBlock> OIL8Encoder::encode_tensor(const Tensor& t, const std::string& name) {
    const float* data = t.data<float>();
    uint32_t total = static_cast<uint32_t>(t.numel());
    uint32_t bs = static_cast<uint32_t>(config_.block_size);

    std::vector<EncodedBlock> blocks;
    uint32_t block_id = 0;
    for (uint32_t offset = 0; offset < total; offset += bs) {
        uint32_t count = std::min(bs, total - offset);
        blocks.push_back(encode(data + offset, count, block_id++));
        blocks.back().name = name;
    }

    return blocks;
}

OIL8Decoder::OIL8Decoder() {}

Tensor OIL8Decoder::decode(const EncodedBlock& block) {
    Tensor out(Shape(static_cast<int64_t>(block.num_weights)), DType::F32);
    float* out_data = out.data<float>();

    for (uint32_t i = 0; i < block.num_weights; i++) {
        out_data[i] = block.codebook.dequantize(block.indices[i]);
    }

    return out;
}

Tensor OIL8Decoder::decode_blocks(const std::vector<EncodedBlock>& blocks, const Shape& original_shape) {
    int64_t total = original_shape.numel();
    Tensor out(Shape(total), DType::F32);
    float* out_data = out.data<float>();

    int64_t offset = 0;
    for (const auto& block : blocks) {
        for (uint32_t i = 0; i < block.num_weights; i++) {
            out_data[offset++] = block.codebook.dequantize(block.indices[i]);
        }
    }

    return out.reshape(original_shape);
}

} // namespace oil8
} // namespace oil
