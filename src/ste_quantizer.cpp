#include "quant/ste_quantizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace quant {

STEQuantizer::STEQuantizer(Format target_format) : target_format_(target_format) {}

void STEQuantizer::set_target_format(Format fmt) {
    target_format_ = fmt;
}

Format STEQuantizer::target_format() const {
    return target_format_;
}

Tensor STEQuantizer::forward(const Tensor& fp32_weight) {
    // Real STE: quantize → dequantize → return
    // Forward pass sees quantization noise
    // Backward (STE): gradient flows through unchanged
    int64_t n = fp32_weight.numel();
    const float* src = (const float*)fp32_weight.data();
    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();

    switch (target_format_) {
        case Format::Q1: {
            std::vector<uint8_t> packed((n + 3) / 4);
            float scale;
            quantize_quant(src, packed.data(), &scale, n);
            for (int64_t i = 0; i < n; i++) {
                int v = (packed[i / 4] >> (2 * (i % 4))) & 3;
                rd[i] = (float)(v == 1 ? 1 : v == 2 ? -1 : 0) * scale;
            }
            break;
        }
        case Format::Q8: {
            static CodebookQUANT8 cb8;
            static bool cb8_trained = false;
            if (!cb8_trained) {
                cb8.train(src, (size_t)n);
                cb8_trained = true;
            }
            for (int64_t i = 0; i < n; i++)
                rd[i] = cb8.dequantize(cb8.quantize(src[i]));
            break;
        }
        case Format::Q4: {
            static CodebookQUANT4 cb4;
            static bool cb4_trained = false;
            if (!cb4_trained) {
                cb4.train(src, (size_t)n);
                cb4_trained = true;
            }
            for (int64_t i = 0; i < n; i++)
                rd[i] = cb4.dequantize(cb4.quantize(src[i]));
            break;
        }
        default:
            std::memcpy(rd, src, (size_t)n * sizeof(float));
            break;
    }
    return result;
}

float STEQuantizer::find_scale(const float* data, int64_t n) {
    float max_abs = 0;
    for (int64_t i = 0; i < n; i++) {
        float a = std::fabs(data[i]);
        if (a > max_abs) max_abs = a;
    }
    return max_abs > 1e-10f ? max_abs : 1.0f;
}

Tensor STEQuantizer::quantize_with_codebook(const Tensor& fp32_weight, CodebookQUANT8& codebook) {
    int64_t n = fp32_weight.numel();
    const float* src = (const float*)fp32_weight.data();

    // Train codebook on the data if centroids are all zero
    bool needs_train = true;
    if (codebook.centroids[0] != 0.0f || codebook.centroids[CodebookQUANT8::SIZE - 1] != 0.0f) {
        needs_train = false;
    }
    if (needs_train) {
        codebook.train(src, (size_t)n);
    }

    Tensor quantized(Shape{n}, DType::U8);
    uint8_t* dst = (uint8_t*)quantized.data();

    for (int64_t i = 0; i < n; i++) {
        dst[i] = codebook.quantize(src[i]);
    }

    // Dequantize back to fp32 for STE (differentiable approximation)
    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();
    for (int64_t i = 0; i < n; i++) {
        rd[i] = codebook.dequantize(dst[i]);
    }

    return result;
}

Tensor STEQuantizer::quantize_with_codebook(const Tensor& fp32_weight, CodebookQUANT4& codebook) {
    int64_t n = fp32_weight.numel();
    const float* src = (const float*)fp32_weight.data();

    bool needs_train = true;
    for (int i = 0; i < CodebookQUANT4::SIZE; i++) {
        if (codebook.centroids[i] != 0) { needs_train = false; break; }
    }
    if (needs_train) {
        codebook.train(src, (size_t)n);
    }

    int packed_bytes = (int)((n + 1) / 2);
    Tensor quantized(Shape{packed_bytes}, DType::U4);
    uint8_t* dst = (uint8_t*)quantized.data();
    std::memset(dst, 0, (size_t)packed_bytes);

    for (int64_t i = 0; i < n; i++) {
        uint8_t idx = codebook.quantize(src[i]);
        if (i % 2 == 0) {
            dst[i / 2] = (dst[i / 2] & 0xF0) | (idx & 0x0F);
        } else {
            dst[i / 2] = (dst[i / 2] & 0x0F) | ((idx & 0x0F) << 4);
        }
    }

    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();
    for (int64_t i = 0; i < n; i++) {
        uint8_t idx;
        if (i % 2 == 0) idx = dst[i / 2] & 0x0F;
        else idx = (dst[i / 2] >> 4) & 0x0F;
        rd[i] = codebook.dequantize(idx);
    }

    return result;
}

Tensor STEQuantizer::forward_mixed(const Tensor& weights, const std::vector<Format>& per_block_formats, int block_size) {
    int64_t n = weights.numel();
    const float* src = (const float*)weights.data();
    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();

    int64_t num_blocks = (int64_t)per_block_formats.size();

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t block_start = b * block_size;
        int64_t block_end = std::min(block_start + block_size, n);
        int64_t block_n = block_end - block_start;
        if (block_n <= 0) break;

        Format fmt = per_block_formats[(size_t)b];

        switch (fmt) {
        case Format::Q1: {
                std::vector<uint8_t> packed((block_n + 3) / 4);
                float scale;
                quantize_quant(src + block_start, packed.data(), &scale, block_n);
                for (int64_t i = 0; i < block_n; i++) {
                    int v = (packed[(size_t)i / 4] >> (2 * (i % 4))) & 3;
                    rd[block_start + i] = (float)(v == 1 ? 1 : v == 2 ? -1 : 0) * scale;
                }
                break;
            }
            case Format::Q1: {
                std::vector<uint8_t> packed((block_n + 7) / 8);
                float scale;
                quantize_Q1(src + block_start, packed.data(), &scale, block_n);
                for (int64_t i = 0; i < block_n; i++) {
                    int v = (packed[(size_t)i / 8] >> (i % 8)) & 1;
                    rd[block_start + i] = (float)(v == 0 ? -1 : 1) * scale;
                }
                break;
            }
            case Format::Q8: {
                CodebookQUANT8 cb;
                cb.train(src + block_start, (size_t)block_n);
                for (int64_t i = 0; i < block_n; i++)
                    rd[block_start + i] = cb.dequantize(cb.quantize(src[block_start + i]));
                break;
            }
            case Format::Q4: {
                CodebookQUANT4 cb;
                cb.train(src + block_start, (size_t)block_n);
                for (int64_t i = 0; i < block_n; i++)
                    rd[block_start + i] = cb.dequantize(cb.quantize(src[block_start + i]));
                break;
            }
            default:
                std::memcpy(rd + block_start, src + block_start, (size_t)block_n * sizeof(float));
                break;
        }
    }
    return result;
}

void STEQuantizer::quantize_quant(const float* src, uint8_t* dst, float* scale, int64_t n) {
    float s = find_scale(src, n);
    float threshold = s * 0.5f;
    *scale = s;

    int packed_size = (int)((n + 3) / 4);
    std::memset(dst, 0, (size_t)packed_size);

    for (int64_t i = 0; i < n; i++) {
        uint8_t val;
        if (src[i] > threshold) {
            val = 1;
        } else if (src[i] < -threshold) {
            val = 2; // -1 mapped to 2 in 2-bit encoding
        } else {
            val = 0;
        }
        dst[i / 4] |= (val & 0x03) << (2 * (i % 4));
    }
}

void STEQuantizer::quantize_Q1(const float* src, uint8_t* dst, float* scale, int64_t n) {
    float s = find_scale(src, n);
    *scale = s;

    int packed_size = (int)((n + 7) / 8);
    std::memset(dst, 0, (size_t)packed_size);

    for (int64_t i = 0; i < n; i++) {
        if (src[i] > 0) {
            dst[i / 8] |= (1 << (i % 8));
        }
    }
}

} // namespace quant
