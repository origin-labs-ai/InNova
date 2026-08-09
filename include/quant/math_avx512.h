#pragma once

#include "quant/tensor.h"

namespace quant {
namespace math {

void dot_avx512(float* result, const float* a, const float* b, int64_t n);
void axpy_avx512(float alpha, const float* x, float* y, int64_t n);

void vec_add_avx512(float* dst, const float* a, const float* b, int64_t n);
void vec_sub_avx512(float* dst, const float* a, const float* b, int64_t n);
void vec_mul_avx512(float* dst, const float* a, const float* b, int64_t n);
void vec_div_avx512(float* dst, const float* a, const float* b, int64_t n);
void vec_scale_avx512(float* dst, const float* src, float factor, int64_t n);

void relu_avx512(float* dst, const float* src, int64_t n);
void gelu_avx512(float* dst, const float* src, int64_t n);
void silu_avx512(float* dst, const float* src, int64_t n);
void sigmoid_avx512(float* dst, const float* src, int64_t n);
void tanh_avx512(float* dst, const float* src, int64_t n);

void softmax_avx512(float* dst, const float* src, int64_t n);
void layer_norm_avx512(float* dst, const float* src, const float* gamma,
                       const float* beta, int64_t n, float eps);
void rms_norm_avx512(float* dst, const float* src, const float* gamma,
                     int64_t n, float eps);

void gemv_avx512(float alpha, const Tensor& A, const Tensor& x,
                 float beta, Tensor& y);

} // namespace math
} // namespace quant
