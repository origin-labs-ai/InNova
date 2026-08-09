#pragma once

#include "quant/tensor.h"

namespace quant {
namespace math {

float dot(const Tensor& a, const Tensor& b);
void axpy(float alpha, const Tensor& x, Tensor& y);
float norm(const Tensor& x);
float asum(const Tensor& x);

void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y);
void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C);

void relu(const Tensor& x, Tensor& y);
void gelu(const Tensor& x, Tensor& y);
void silu(const Tensor& x, Tensor& y);
void sigmoid(const Tensor& x, Tensor& y);
void tanh_(const Tensor& x, Tensor& y);

void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y);
void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y);

void softmax(const Tensor& x, Tensor& y, int axis = -1);

void add(const Tensor& a, const Tensor& b, Tensor& c);
void sub(const Tensor& a, const Tensor& b, Tensor& c);
void mul(const Tensor& a, const Tensor& b, Tensor& c);
void scale(float s, const Tensor& x, Tensor& y);

float mean(const Tensor& x);
float sum(const Tensor& x);
float max(const Tensor& x);

Tensor zeros_like(const Tensor& x);
Tensor ones_like(const Tensor& x);

// ===========================================================================
// SIMD vector math functions — raw pointer interface for performance
// ===========================================================================

void vec_exp(float* dst, const float* src, int n);
void vec_log(float* dst, const float* src, int n);
void vec_sigmoid(float* dst, const float* src, int n);
void vec_tanh(float* dst, const float* src, int n);
void vec_pow(float* dst, const float* src, float exp_val, int n);
void vec_erf(float* dst, const float* src, int n);
void vec_softmax_stable(float* dst, const float* src, int n);
void vec_layer_norm(float* dst, const float* src, const float* gamma, const float* beta, int n, float eps);
void vec_rms_norm(float* dst, const float* src, const float* gamma, int n, float eps);

void vec_exp_avx2(float* dst, const float* src, int n);
void vec_log_avx2(float* dst, const float* src, int n);
void vec_sigmoid_avx2(float* dst, const float* src, int n);
void vec_tanh_avx2(float* dst, const float* src, int n);
void vec_pow_avx2(float* dst, const float* src, float exp_val, int n);
void vec_erf_avx2(float* dst, const float* src, int n);

void vec_relu(float* dst, const float* src, int n);
void vec_gelu(float* dst, const float* src, int n);
void vec_silu(float* dst, const float* src, int n);
void vec_add(float* dst, const float* a, const float* b, int n);
void vec_sub(float* dst, const float* a, const float* b, int n);
void vec_mul(float* dst, const float* a, const float* b, int n);
void vec_scale(float* dst, const float* src, float factor, int n);
void vec_negate(float* dst, const float* src, int n);
void vec_abs(float* dst, const float* src, int n);
void vec_sqrt(float* dst, const float* src, int n);
void vec_dot(float* result, const float* a, const float* b, int n);
void vec_norm(float* result, const float* src, int n);
void vec_sum(float* result, const float* src, int n);
void vec_maximum(float* dst, const float* a, const float* b, int n);
void vec_minimum(float* dst, const float* a, const float* b, int n);
void vec_clip(float* dst, const float* src, float lo, float hi, int n);
void vec_lerp(float* dst, const float* a, const float* b, float t, int n);
void vec_smoothstep(float* dst, const float* src, float edge0, float edge1, int n);
void vec_fp32_to_fp16(uint16_t* dst, const float* src, int n);
void vec_fp16_to_fp32(float* dst, const uint16_t* src, int n);
void vec_softmax_stable_avx2(float* dst, const float* src, int n);
void vec_layer_norm_avx2(float* dst, const float* src, const float* gamma, const float* beta, int n, float eps);
void vec_rms_norm_avx2(float* dst, const float* src, const float* gamma, int n, float eps);

} // namespace math
} // namespace quant
