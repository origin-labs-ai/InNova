#pragma once
#include "quant/autograd.h"
#include "quant/math.h"
#include "quant/kernel.h"
#include <cmath>
#include <cstring>
#include <cfloat>

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(QUANT_AVX2) || defined(__AVX2__)
static inline __m256 quant_internal_mm256_exp_ps(__m256 v) {
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, v);
    for (int i = 0; i < 8; i++) tmp[i] = std::exp(tmp[i]);
    return _mm256_loadu_ps(tmp);
}
#if !defined(_mm256_exp_ps)
#define _mm256_exp_ps(v) quant_internal_mm256_exp_ps(v)
#endif
#endif

namespace quant {

// ========================================================================
// Softmax kernel (AVX2 accelerated) — used by ScaledDotProductAttentionFunction
// ========================================================================

static inline void softmax_forward_avx2(float* output, const float* input, int64_t cols) {
    int64_t i;
    float max_val = -FLT_MAX;
    for (i = 0; i < cols; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum_exp = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    __m256 vmax = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    for (i = 0; i + 8 <= cols; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        v = _mm256_sub_ps(v, vmax);
        v = _mm256_exp_ps(v);
        _mm256_storeu_ps(output + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, vsum);
    for (int j = 0; j < 8; j++) sum_exp += sum_arr[j];
#else
    for (i = 0; i < cols; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }
#endif
    for (; i < cols; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }

    float inv_sum = 1.0f / (sum_exp + 1e-10f);
#if defined(QUANT_AVX2) || defined(__AVX2__)
    __m256 v_inv = _mm256_set1_ps(inv_sum);
    for (i = 0; i + 8 <= cols; i += 8) {
        __m256 v = _mm256_loadu_ps(output + i);
        v = _mm256_mul_ps(v, v_inv);
        _mm256_storeu_ps(output + i, v);
    }
#endif
    for (; i < cols; i++) {
        output[i] *= inv_sum;
    }
}

// ========================================================================
// ScaledDotProductAttentionFunction: Q @ K^T / sqrt(D) -> softmax -> @ V
// Q: {B, H, S, D}, K/V: {B, KV_H, S_full, D}
// ========================================================================

class ScaledDotProductAttentionFunction : public AutogradFunction {
public:
    ScaledDotProductAttentionFunction(int64_t nh, int64_t nkv, int64_t hd)
        : num_heads_(nh), num_kv_heads_(nkv), head_dim_(hd),
          B_(0), H_(0), S_(0), D_(0), KV_H_(0), S_full_(0) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;

private:
    int64_t num_heads_, num_kv_heads_, head_dim_;
    int64_t B_, H_, S_, D_, KV_H_, S_full_;
};

// ========================================================================
// MatMulFunction
// ========================================================================

class MatMulFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
};

// ========================================================================
// AddFunction
// ========================================================================

class AddFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
};

// ========================================================================
// SiLUFunction
// ========================================================================

class SiLUFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
};

// ========================================================================
// MulFunction
// ========================================================================

class MulFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
};

// ========================================================================
// RMSNormFunction
// ========================================================================

class RMSNormFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
    float saved_eps = 1e-5f;
};

// ========================================================================
// CrossEntropyFunction
// ========================================================================

class CrossEntropyFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
};

// ========================================================================
// RotaryFunction: applies RoPE rotation (differentiable)
// ========================================================================

class RotaryFunction : public AutogradFunction {
public:
    RotaryFunction(int64_t hd, const Tensor& cos, const Tensor& sin,
                   int64_t ss, int64_t sl);
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;

private:
    int64_t head_dim_;
    Tensor cos_cached_, sin_cached_;
    int64_t seq_start_, seq_len_;
    int64_t B_, H_, S_, D_;
};

// ========================================================================
// BiasAddFunction: out = x + bias (broadcast over dim 0)
// ========================================================================

class BiasAddFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
};

// ========================================================================
// EmbeddingFunction: embedding lookup (differentiable w.r.t. weight)
// ========================================================================

class EmbeddingFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;
};

// ========================================================================
// FlattenAttentionFunction: {B,H,S,D} -> {B*S, H*D} with data reordering
// ========================================================================

class FlattenAttentionFunction : public AutogradFunction {
public:
    FlattenAttentionFunction(int64_t B, int64_t H, int64_t S, int64_t D);
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;

private:
    int64_t B_, H_, S_, D_;
};

// ========================================================================
// TransposeFunction: permute two dims (differentiable)
// ========================================================================

class TransposeFunction : public AutogradFunction {
public:
    TransposeFunction(int dim1, int dim2);
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override;

private:
    int dim1_, dim2_;
};

} // namespace quant

#if defined(_mm256_exp_ps)
#undef _mm256_exp_ps
#endif
