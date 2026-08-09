#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/codebook.h"

namespace quant {

// Straight-Through Estimator for QUANT-native training
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
    Tensor quantize_with_codebook(const Tensor& fp32_weight, CodebookQUANT8& codebook);
    Tensor quantize_with_codebook(const Tensor& fp32_weight, CodebookQUANT4& codebook);
    
    // Quantize to QUANT/Q1 with scale
    void quantize_quant(const float* src, uint8_t* dst, float* scale, int64_t n);
    void quantize_Q1(const float* src, uint8_t* dst, float* scale, int64_t n);
    
    // Set target format
    void set_target_format(Format fmt);
    Format target_format() const;
    
private:
    Format target_format_ = Format::Q1;
    
    // Find scale factor (max abs)
    float find_scale(const float* data, int64_t n);
};

} // namespace quant
