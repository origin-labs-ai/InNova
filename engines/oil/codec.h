#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "oil/tensor.h"
#include "oil/codebook.h"

namespace oil {
namespace oil8 {

struct CodecConfig {
    int block_size = 256;
    int codebook_size = 256;
    int train_iters = 20;
    bool parallel = true;
};

struct EncodedBlock {
    uint32_t block_id;
    std::string name;
    CodebookOIL8 codebook;
    std::vector<uint8_t> indices;
    uint32_t num_weights;
};

class OIL8Encoder {
public:
    explicit OIL8Encoder(const CodecConfig& cfg = CodecConfig());
    EncodedBlock encode(const float* weights, uint32_t num_weights, uint32_t block_id);
    std::vector<EncodedBlock> encode_tensor(const Tensor& t, const std::string& name);
    CodecConfig config() const;
private:
    CodecConfig config_;
};

class OIL8Decoder {
public:
    OIL8Decoder();
    Tensor decode(const EncodedBlock& block);
    Tensor decode_blocks(const std::vector<EncodedBlock>& blocks, const Shape& original_shape);
};

} // namespace oil8
} // namespace oil
