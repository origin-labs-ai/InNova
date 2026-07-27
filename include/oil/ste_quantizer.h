#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/codebook.h"

namespace oil {

// Straight-Through Estimator for OIL-native training
class STEQuantizer {
public:
    STEQuantizer() = default;
    explicit STEQuantizer(Format target_format);
    
    // Forward: quantize weights to target format
    // Backward: identity gradient (straight-through)
    Tensor forward(const Tensor& fp32_weight);

    // Mixed-format forward: different formats per block
    // per_block_formats[i] specifies the format for the i-th block of size block_size
    // The last block may be smaller than block_size
    Tensor forward_mixed(const Tensor& weights, const std::vector<Format>& per_block_formats, int block_size = 256);
    
    // Quantize with codebook training
    Tensor quantize_with_codebook(const Tensor& fp32_weight, CodebookOIL8& codebook);
    Tensor quantize_with_codebook(const Tensor& fp32_weight, CodebookOIL4& codebook);
    
    // Quantize to SPARK/OIL1 with scale
    void quantize_spark(const float* src, uint8_t* dst, float* scale, int64_t n);
    void quantize_oil1(const float* src, uint8_t* dst, float* scale, int64_t n);
    
    // Set target format
    void set_target_format(Format fmt);
    Format target_format() const;
    
private:
    Format target_format_ = Format::SPARK_Q0;
    
    // Find scale factor (max abs)
    float find_scale(const float* data, int64_t n);
};

} // namespace oil
