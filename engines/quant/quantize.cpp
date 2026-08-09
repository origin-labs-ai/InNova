#include "quantize.h"
#include "quant/tensor.h"
#include "quant/types.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace quant {
namespace quant8 {

QuantizeParams quantize_tensor(const Tensor& src, Tensor& dst) {
    QUANT_CHECK(src.dtype() == DType::F32, "quantize_tensor: src must be F32");
    QUANT_CHECK(src.numel() == dst.numel(), "quantize_tensor: size mismatch");

    const float* src_data = src.data<float>();
    int64_t n = src.numel();

    float min_v = 0.0f, max_v = 0.0f;
    if (n > 0) {
        min_v = max_v = src_data[0];
        for (int64_t i = 1; i < n; i++) {
            min_v = std::min(min_v, src_data[i]);
            max_v = std::max(max_v, src_data[i]);
        }
    }
    // Asymmetric affine: q = round(x / inv_scale + zero_point), x in [min, max] -> [0, 255]
    float range = max_v - min_v;
    if (range < 1e-12f) range = 1.0f;

    QuantizeParams params;
    params.inv_scale = range / 255.0f;
    params.scale = 1.0f / params.inv_scale;
    params.zero_point = -min_v * params.scale; // so min maps near 0

    if (dst.dtype() == DType::U8) {
        uint8_t* dst_data = dst.data<uint8_t>();
        for (int64_t i = 0; i < n; i++) {
            float q = std::round(src_data[i] * params.scale + params.zero_point);
            dst_data[i] = static_cast<uint8_t>(std::clamp(q, 0.0f, 255.0f));
        }
    } else if (dst.dtype() == DType::F32) {
        // Keep scaled representation without zero-point for F32 dest (legacy path)
        float* dst_data = dst.data<float>();
        float max_abs = std::max(std::abs(min_v), std::abs(max_v));
        if (max_abs < 1e-12f) max_abs = 1.0f;
        params.scale = 127.0f / max_abs;
        params.inv_scale = max_abs / 127.0f;
        params.zero_point = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            dst_data[i] = src_data[i] * params.scale;
        }
    } else {
        QUANT_CHECK(false, "quantize_tensor: unsupported dst dtype");
    }

    return params;
}

std::vector<QuantizeParams> quantize_activations(const Tensor& src, Tensor& dst) {
    QUANT_CHECK(src.dtype() == DType::F32, "quantize_activations: src must be F32");
    QUANT_CHECK(src.rank() >= 2, "quantize_activations: src must have rank >= 2");

    int64_t batch = src.dim(0);
    int64_t per_batch = src.numel() / batch; (void)per_batch;

    std::vector<QuantizeParams> params(batch);

    for (int64_t b = 0; b < batch; b++) {
        Tensor src_slice = src.slice(0, b, b + 1);
        Tensor dst_slice = dst.slice(0, b, b + 1);
        params[b] = quantize_tensor(src_slice, dst_slice);
    }

    return params;
}

Tensor dequantize(const Tensor& src, const QuantizeParams& params) {
    int64_t n = src.numel();
    Tensor out(Shape(n), DType::F32);
    float* out_data = out.data<float>();

    if (src.dtype() == DType::U8) {
        const uint8_t* src_data = src.data<uint8_t>();
        for (int64_t i = 0; i < n; i++) {
            // Asymmetric dequant: (q - zero_point) * inv_scale
            out_data[i] = (static_cast<float>(src_data[i]) - params.zero_point) * params.inv_scale;
        }
    } else if (src.dtype() == DType::F32) {
        const float* src_data = src.data<float>();
        for (int64_t i = 0; i < n; i++) {
            out_data[i] = src_data[i] * params.inv_scale;
        }
    } else {
        QUANT_CHECK(false, "dequantize: unsupported src dtype");
    }

    return out;
}



} // namespace quant8
} // namespace quant
