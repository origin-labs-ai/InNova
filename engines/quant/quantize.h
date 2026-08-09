#pragma once
#include <cstdint>
#include <vector>
#include "quant/tensor.h"

namespace quant {
namespace quant8 {

struct QuantizeParams {
    float scale = 1.0f;
    float inv_scale = 1.0f;
    float zero_point = 0.0f; // asymmetric U8: real = (q - zero_point) * inv_scale
};

QuantizeParams quantize_tensor(const Tensor& src, Tensor& dst);
std::vector<QuantizeParams> quantize_activations(const Tensor& src, Tensor& dst);
Tensor dequantize(const Tensor& src, const QuantizeParams& params);


} // namespace quant8
} // namespace quant
