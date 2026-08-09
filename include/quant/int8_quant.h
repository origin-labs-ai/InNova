#pragma once
#include "quant/types.h"
#include "quant/tensor.h"

namespace quant {

struct Int8QuantParams {
    float scale;   // multiplier: int8 = round(fp32 * scale)
    float inv_scale; // 1/scale for dequant
};

// Per-tensor INT8 quantization
// Finds max absolute value across all elements
// Scale = 127 / max_abs
// Returns quantized data and scale
Int8QuantParams quantize_per_tensor(const float* src, int8_t* dst, int64_t n);

// Per-token (per-row) INT8 quantization
// Each row gets its own scale
std::vector<Int8QuantParams> quantize_per_token(const float* src, int8_t* dst,
                                                  int64_t rows, int64_t cols);

// Dequantize: float = int8 * inv_scale
void dequantize_per_tensor(const int8_t* src, float* dst, 
                           int64_t n, float inv_scale);

// Quantize Tensor (convenience)
Int8QuantParams quantize_tensor(const Tensor& src, Tensor& dst);

} // namespace quant
