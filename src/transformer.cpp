#include "quant/transformer.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/random.h"
#include "quant/simd_math.h"
#include "quant/flash_attention.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <utility>

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace quant {

// Initialize weight with uniform random values in [-bound, bound]
static void init_uniform(Tensor& t, float bound, int block_idx = 0) {
    RNG rng(42 + block_idx * 7919);
    float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++)
        d[i] = (rng.uniform() * 2.0f - 1.0f) * bound;
}

// Embedding
Embedding::Embedding(int64_t vocab_size, int64_t dim)
    : weight(Tensor::zeros(Shape{vocab_size, dim})) {
    float scale = 1.0f / std::sqrt((float)dim);
    init_uniform(weight, scale);
}

Tensor Embedding::forward(const Tensor& input_ids) const {
    if (AutogradEngine::enabled())
        return AutogradEngine::embedding_op(input_ids, weight);
    int64_t batch = input_ids.numel();
    int64_t dim = weight.shape().dims[1];
    int64_t vocab = weight.shape().dims[0];
    Tensor out(Shape{batch, dim}, DType::F32);
    const float* ids = input_ids.data<float>();
    const float* w = weight.data<float>();
    float* od = out.data<float>();
    for (int64_t i = 0; i < batch; i++) {
        int64_t id = (int64_t)ids[i];
        if (id < 0) id = 0;
        if (id >= vocab) id = 0;
        memcpy(od + i * dim, w + id * dim, dim * sizeof(float));
    }
    return out;
}

size_t Embedding::param_count() const { return weight.numel(); }

// Linear
Linear::Linear(int64_t in_features, int64_t out_features)
    : weight(Tensor::zeros(Shape{out_features, in_features})),
      bias(Tensor::zeros(Shape{out_features})) {
    float scale = 1.0f / std::sqrt((float)in_features);
    init_uniform(weight, scale);
}

Tensor Linear::forward(const Tensor& input) const {
    int64_t in_dim = weight.shape().dims[1];
    int64_t out_dim = weight.shape().dims[0];
    int64_t in_rank = input.rank();
    int64_t batch = input.numel() / in_dim;
    
    Tensor inp2d = (in_rank > 2) ? input.reshape(Shape{batch, in_dim}) : input;
    
    Tensor out = AutogradEngine::matmul_op(inp2d, weight, batch, out_dim, in_dim);
    
    if (bias.numel() > 0) {
        if (AutogradEngine::enabled()) {
            out = AutogradEngine::bias_add_op(out, bias);
        } else {
            float* od = (float*)out.data();
            const float* bd = (const float*)bias.data();
            for (int64_t i = 0; i < batch; i++) {
                float* row = od + i * out_dim;
                int64_t j = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
                for (; j + 8 <= out_dim; j += 8) {
                    __m256 ov = _mm256_loadu_ps(row + j);
                    __m256 bv = _mm256_loadu_ps(bd + j);
                    _mm256_storeu_ps(row + j, _mm256_add_ps(ov, bv));
                }
#endif
                for (; j < out_dim; j++)
                    row[j] += bd[j];
            }
        }
    }
    
    // Restore leading dims if input was > 2D
    if (in_rank > 2) {
        Shape out_shape = input.shape();
        out_shape.dims[in_rank - 1] = out_dim;
        return out.reshape(out_shape);
    }
    return out;
}

size_t Linear::param_count() const { return weight.numel() + bias.numel(); }

// RMSNorm
RMSNorm::RMSNorm(int64_t size, float eps_val)
    : weight(Tensor::ones(Shape{size})), eps(eps_val) {}

Tensor RMSNorm::forward(const Tensor& input) const {
    return AutogradEngine::rms_norm_op(input, weight, eps);
}

// RotaryEmbedding
RotaryEmbedding::RotaryEmbedding(int64_t hd, int64_t max_seq_len, float t)
    : head_dim(hd), theta(t) {
    cos_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    sin_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    float* cos_d = (float*)cos_cached.data();
    float* sin_d = (float*)sin_cached.data();
    for (int64_t i = 0; i < max_seq_len; i++) {
        for (int64_t j = 0; j < hd / 2; j++) {
            float inv_freq = 1.0f / std::pow(theta, (float)(2 * j) / hd);
            float val = (float)i * inv_freq;
            cos_d[i * hd / 2 + j] = std::cos(val);
            sin_d[i * hd / 2 + j] = std::sin(val);
        }
    }
}

// YARN / NTK-aware RoPE — long-context extension
RotaryEmbedding::RotaryEmbedding(int64_t hd, int64_t max_seq_len, float t,
                                 RoPEScalingMode mode, float factor,
                                 int64_t original_max_seq_len,
                                 float yarn_beta_fast,
                                 float yarn_beta_slow,
                                 float /*yarn_attn_factor*/)
    : head_dim(hd), theta(t), scaling_mode(mode), scaling_factor(factor) {
    cos_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    sin_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    float* cos_d = (float*)cos_cached.data();
    float* sin_d = (float*)sin_cached.data();

    int64_t orig_len = original_max_seq_len > 0 ? original_max_seq_len : max_seq_len;
    float inter_len = (mode == RoPEScalingMode::YARN || mode == RoPEScalingMode::Linear)
                      ? (float)orig_len : (float)max_seq_len;

    for (int64_t i = 0; i < max_seq_len; i++) {
        for (int64_t j = 0; j < hd / 2; j++) {
            float inv_freq_base = 1.0f / std::pow(theta, (float)(2 * j) / hd);
            float freq = inv_freq_base;
            float v_pos = (float)i;

            if (mode == RoPEScalingMode::Linear) {
                v_pos /= factor;
            } else if (mode == RoPEScalingMode::NTK) {
                float alpha = factor;
                float base = std::pow(alpha, (float)hd / (hd - 2.0f));
                float ntK_theta = theta * base;
                float ntK_inv_freq = 1.0f / std::pow(ntK_theta, (float)(2 * j) / hd);
                freq = ntK_inv_freq;
            } else if (mode == RoPEScalingMode::YARN) {
                float w = j < hd / 2 ? (float)j / (hd / 2) : 1.0f;
                float ext_f = 1.0f / factor;
                float mscale = 1.0f;
                if (w < yarn_beta_slow / yarn_beta_fast) {
                    freq = inv_freq_base;
                } else if (w > 1.0f) {
                    freq = inv_freq_base * ext_f;
                } else {
                    float smooth = (w - yarn_beta_slow / yarn_beta_fast) / (1.0f - yarn_beta_slow / yarn_beta_fast);
                    smooth = 0.5f * (1.0f - std::cos(smooth * 3.14159265f));
                    freq = inv_freq_base * ((1.0f - smooth) + smooth * ext_f);
                }
                (void)mscale; (void)inter_len;
            }

            float val = v_pos * freq;
            cos_d[i * hd / 2 + j] = std::cos(val);
            sin_d[i * hd / 2 + j] = std::sin(val);
        }
    }
}

void RotaryEmbedding::apply(Tensor& x, int64_t seq_start, int64_t seq_len) const {
    int64_t B = x.shape().dims[0];
    int64_t H = x.shape().dims[1];
    int64_t S = x.shape().dims[2];
    int64_t D = x.shape().dims[3];
    int64_t half_D = D / 2;
    float* xd = (float*)x.data();
    const float* cos_d = (const float*)cos_cached.data();
    const float* sin_d = (const float*)sin_cached.data();
    int64_t HS = H * S;
    int64_t HSD = HS * D;
    int64_t cos_stride = cos_cached.shape().dims[1];
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                int64_t pos = seq_start + s;
                int64_t x_off = b * HSD + h * S * D + s * D;
                int64_t c_off = pos * cos_stride;
                int64_t d = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
                for (; d + 8 <= half_D; d += 8) {
                    __m256 x1v = _mm256_loadu_ps(xd + x_off + d);
                    __m256 x2v = _mm256_loadu_ps(xd + x_off + d + half_D);
                    __m256 cosv = _mm256_loadu_ps(cos_d + c_off + d);
                    __m256 sinv = _mm256_loadu_ps(sin_d + c_off + d);
                    _mm256_storeu_ps(xd + x_off + d,
                        _mm256_sub_ps(_mm256_mul_ps(x1v, cosv), _mm256_mul_ps(x2v, sinv)));
                    _mm256_storeu_ps(xd + x_off + d + half_D,
                        _mm256_add_ps(_mm256_mul_ps(x1v, sinv), _mm256_mul_ps(x2v, cosv)));
                }
#endif
                for (; d < half_D; d++) {
                    float x1 = xd[x_off + d];
                    float x2 = xd[x_off + d + half_D];
                    float cv = cos_d[c_off + d];
                    float sv = sin_d[c_off + d];
                    xd[x_off + d] = x1 * cv - x2 * sv;
                    xd[x_off + d + half_D] = x1 * sv + x2 * cv;
                }
            }
        }
    }
}

// Attention
Attention::Attention(const TransformerConfig& cfg)
    : num_heads(cfg.num_heads), 
      num_kv_heads(cfg.num_kv_heads > 0 ? cfg.num_kv_heads : cfg.num_heads),
      head_dim(cfg.head_dim)
{
    if (cfg.rope_scaling_mode != RoPEScalingMode::None) {
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta,
                               cfg.rope_scaling_mode, cfg.rope_scaling_factor,
                               cfg.rope_original_max_seq_len,
                               cfg.yarn_beta_fast, cfg.yarn_beta_slow,
                               cfg.yarn_attn_factor);
    } else {
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta);
    }
    q_proj = Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim);
    k_proj = Linear(cfg.hidden_size, num_kv_heads * cfg.head_dim);
    v_proj = Linear(cfg.hidden_size, num_kv_heads * cfg.head_dim);
    o_proj = Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
}

Tensor Attention::forward(const Tensor& x, const Tensor& positions,
                           const Tensor& mask, KVCache& cache, int layer_idx) const {
    int64_t B = x.shape().dims[0];
    int64_t S = x.shape().dims[1];
    
    Tensor q = q_proj.forward(x);
    Tensor k = k_proj.forward(x);
    Tensor v = v_proj.forward(x);
    
    Tensor q_reshaped = q.reshape(Shape{B, S, num_heads, head_dim});
    Tensor k_reshaped = k.reshape(Shape{B, S, num_kv_heads, head_dim});
    Tensor v_reshaped = v.reshape(Shape{B, S, num_kv_heads, head_dim});
    
    Tensor attn_out;
    if (AutogradEngine::enabled()) {
        Tensor q_t = AutogradEngine::transpose_op(q_reshaped, 1, 2);
        Tensor k_t = AutogradEngine::transpose_op(k_reshaped, 1, 2);
        Tensor v_t = AutogradEngine::transpose_op(v_reshaped, 1, 2);
        Tensor q_rope = AutogradEngine::rotary_op(q_t, rope.cos_cached, rope.sin_cached, 0, S);
        Tensor k_rope = AutogradEngine::rotary_op(k_t, rope.cos_cached, rope.sin_cached, 0, S);
        attn_out = AutogradEngine::attention_op(q_rope, k_rope, v_t, num_heads, num_kv_heads, head_dim);
    } else {
        Tensor q_t = q_reshaped.transpose(1, 2);  // {B, H, S, D}
        Tensor k_t = k_reshaped.transpose(1, 2);  // {B, KV_H, S, D}
        Tensor v_t = v_reshaped.transpose(1, 2);  // {B, KV_H, S, D}
        
        int64_t seq_start = 0;
        int64_t S_full = S;
        Tensor k_used, v_used;
        if (B > 1) {
            // Multi-batch inputs: cache is batch-1; do full-window causal attention
            rope.apply(q_t, 0, S);
            rope.apply(k_t, 0, S);
            k_used = k_t;
            v_used = v_t;
        } else {
            seq_start = cache.context_len(layer_idx);
            rope.apply(q_t, seq_start, S);
            rope.apply(k_t, seq_start, S);
            cache.append(layer_idx, k_t, v_t);
            auto [k_full, v_full] = cache.get_all(layer_idx);
            k_used = k_full;
            v_used = v_full;
            S_full = k_full.shape().dims[2];
        }
        float scale = 1.0f / std::sqrt((float)head_dim);

        // For GQA: expand K/V from {B, KV_H, S_full, D} to {B, H, S, D}
        Tensor k_expanded, v_expanded;
        if (num_kv_heads < num_heads) {
            int64_t group_size = num_heads / num_kv_heads;
            k_expanded = Tensor(Shape{B, num_heads, S_full, head_dim}, DType::F32);
            v_expanded = Tensor(Shape{B, num_heads, S_full, head_dim}, DType::F32);
            for (int64_t b = 0; b < B; b++) {
                for (int64_t h = 0; h < num_heads; h++) {
                    int64_t kh = h / group_size;
                    for (int64_t s = 0; s < S_full; s++) {
                        int64_t src_off = ((b * num_kv_heads + kh) * S_full + s) * head_dim;
                        int64_t dst_off = ((b * num_heads + h) * S_full + s) * head_dim;
                        memcpy(k_expanded.data<float>() + dst_off,
                               k_used.data<float>() + src_off, head_dim * sizeof(float));
                        memcpy(v_expanded.data<float>() + dst_off,
                               v_used.data<float>() + src_off, head_dim * sizeof(float));
                    }
                }
            }
        } else {
            k_expanded = k_used;
            v_expanded = v_used;
        }

        // Use FlashAttention for long sequences (memory O(n) vs O(n²))
        if (S_full > 64) {
            Tensor causal_mask(Shape{1, 1, S, S_full}, DType::F32);
            float* md = causal_mask.data<float>();
            for (int64_t s = 0; s < S; s++) {
                for (int64_t t = 0; t < S_full; t++) {
                    md[s * S_full + t] = (t > s + seq_start) ? -INFINITY : 0.0f;
                }
            }
            attn_out = flash_attention_forward(q_t, k_expanded, v_expanded,
                                               causal_mask, 0.0f, true);
        } else {
            // Short sequence: use standard attention (simpler, no block overhead)
            const float* qd = (const float*)q_t.data();
            const float* kd = (const float*)k_expanded.data();
            const float* vd = (const float*)v_expanded.data();
            
            Tensor score(Shape{B, num_heads, S, S_full}, DType::F32);
            float* sd = (float*)score.data();
            for (int64_t b = 0; b < B; b++) {
                for (int64_t h = 0; h < num_heads; h++) {
                    int64_t q_base = ((b * num_heads + h) * S) * head_dim;
                    int64_t k_base = ((b * num_heads + h)) * S_full * head_dim;
                    int64_t s_base = (b * num_heads + h) * S * S_full;
                    for (int64_t s = 0; s < S; s++) {
                        const float* qptr = qd + q_base + s * head_dim;
                        for (int64_t t = 0; t < S_full; t++) {
                            const float* kptr = kd + k_base + t * head_dim;
                            float sum = 0;
                            for (int64_t d = 0; d < head_dim; d++)
                                sum += qptr[d] * kptr[d];
                            sd[s_base + s * S_full + t] = sum * scale;
                            if (t > s + seq_start)
                                sd[s_base + s * S_full + t] = -INFINITY;
                        }
                    }
                }
            }
            
            Tensor attn_weights(score.shape(), DType::F32);
            float* wd = (float*)attn_weights.data();
            for (int64_t b = 0; b < B; b++) {
                for (int64_t h = 0; h < num_heads; h++) {
                    int64_t base = (b * num_heads + h) * S * S_full;
                    for (int64_t s = 0; s < S; s++) {
                        int64_t row = base + s * S_full;
                        float max_v = -1e30f;
                        for (int64_t t = 0; t < S_full; t++)
                            if (sd[row + t] > max_v) max_v = sd[row + t];
                        float sum_exp = 0;
                        for (int64_t t = 0; t < S_full; t++) {
                            float e = std::exp(sd[row + t] - max_v);
                            wd[row + t] = e;
                            sum_exp += e;
                        }
                        float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
                        for (int64_t t = 0; t < S_full; t++)
                            wd[row + t] *= inv_sum;
                    }
                }
            }
            
            attn_out = Tensor(Shape{B, num_heads, S, head_dim}, DType::F32);
            float* aod = (float*)attn_out.data();
            for (int64_t b = 0; b < B; b++) {
                for (int64_t h = 0; h < num_heads; h++) {
                    int64_t w_base = (b * num_heads + h) * S * S_full;
                    int64_t v_base = (b * num_heads + h) * S_full * head_dim;
                    int64_t o_base = ((b * num_heads + h) * S) * head_dim;
                    for (int64_t s = 0; s < S; s++) {
                        for (int64_t d = 0; d < head_dim; d++) {
                            float sum = 0;
                            const float* wptr = wd + w_base + s * S_full;
                            const float* vptr = vd + v_base + d;
                            for (int64_t t = 0; t < S_full; t++)
                                sum += wptr[t] * vptr[t * head_dim];
                            aod[o_base + s * head_dim + d] = sum;
                        }
                    }
                }
            }
        }
    }
    
    // Output projection: flatten {B,H,S,D} -> {B*S, H*D}
    Tensor attn_flat;
    if (AutogradEngine::enabled()) {
        attn_flat = AutogradEngine::flatten_attention_op(attn_out, B, num_heads, S, head_dim);
    } else {
        // Transpose {B,H,S,D} -> {B,S,H,D} then flatten
        Tensor attn_t = attn_out.transpose(1, 2);
        attn_flat = attn_t.reshape(Shape{B * S, num_heads * head_dim});
    }
    Tensor o_out = o_proj.forward(attn_flat);
    return o_out.reshape(Shape{B, S, o_out.dim(1)});
}

// FFN
FFN::FFN(const TransformerConfig& cfg)
    : activation(cfg.activation)
{
    gate_proj = Linear(cfg.hidden_size, cfg.ffn_hidden_size);
    up_proj = Linear(cfg.hidden_size, cfg.ffn_hidden_size);
    down_proj = Linear(cfg.ffn_hidden_size, cfg.hidden_size);
}

Tensor FFN::forward(const Tensor& x) const {
    Tensor gate = gate_proj.forward(x);
    Tensor up = up_proj.forward(x);
    
    Tensor hidden;
    if (activation == Activation::SwiGLU || activation == Activation::SiLU) {
        // SwiGLU: silu(gate) * up  — SIMD dispatch
        if (activation == Activation::SwiGLU) {
            hidden = Tensor(gate.shape(), DType::F32);
            const float* gd = gate.data<float>();
            const float* ud = up.data<float>();
            float* hd = hidden.data<float>();
            int64_t n = gate.numel();
            for (int64_t i = 0; i < n; i++) {
                float g = gd[i];
                float sil = g / (1.0f + std::exp(-g));
                hd[i] = sil * ud[i];
            }
        } else {
            gate = AutogradEngine::silu_op(gate);
            hidden = AutogradEngine::mul_op(gate, up);
        }
    } else if (activation == Activation::GeGLU || activation == Activation::GELU) {
        if (activation == Activation::GeGLU) {
            hidden = Tensor(gate.shape(), DType::F32);
            const float* gd = gate.data<float>();
            const float* ud = up.data<float>();
            float* hd = hidden.data<float>();
            int64_t n = gate.numel();
            const float s = 0.7071067811865475f;
            for (int64_t i = 0; i < n; i++) {
                float g = gd[i];
                float gel = 0.5f * g * (1.0f + std::erf(g * s));
                hd[i] = gel * ud[i];
            }
        } else {
            math::gelu(gate, gate);
            hidden = Tensor(gate.shape(), DType::F32);
            const float* gd = gate.data<float>();
            const float* ud = up.data<float>();
            float* hd = hidden.data<float>();
            for (int64_t i = 0; i < gate.numel(); i++) hd[i] = gd[i] * ud[i];
        }
    } else {
        math::relu(gate, gate);
        hidden = Tensor(gate.shape(), DType::F32);
        const float* gd = gate.data<float>();
        const float* ud = up.data<float>();
        float* hd = hidden.data<float>();
        for (int64_t i = 0; i < gate.numel(); i++) hd[i] = gd[i] * ud[i];
    }
    
    return down_proj.forward(hidden);
}

// TransformerBlock
TransformerBlock::TransformerBlock(const TransformerConfig& cfg)
    : attention_norm(cfg.hidden_size, cfg.norm_eps),
      attention(cfg),
      ffn_norm(cfg.hidden_size, cfg.norm_eps),
      ffn(cfg),
      use_parallel_residual(cfg.use_parallel_residual) {}

Tensor TransformerBlock::forward(const Tensor& x, const Tensor& positions,
                                  const Tensor& mask, KVCache& cache, int layer_idx) const {
    if (use_parallel_residual) {
        // GPT-NeoX parallel residual: both attention and FFN see the same pre-norm input
        // output = x + attn(norm1(x)) + ffn(norm2(x))
        Tensor normed_attn = attention_norm.forward(x);
        Tensor normed_ffn  = ffn_norm.forward(x);
        Tensor attn_out = attention.forward(normed_attn, positions, mask, cache, layer_idx);
        Tensor ffn_out  = ffn.forward(normed_ffn);
        Tensor combined = AutogradEngine::add_op(attn_out, ffn_out);
        return AutogradEngine::add_op(combined, x);
    } else {
        // Standard sequential residual (GPT-2/LLaMA style)
        Tensor attn_input = attention_norm.forward(x);
        Tensor attn_out = attention.forward(attn_input, positions, mask, cache, layer_idx);
        attn_out = AutogradEngine::add_op(attn_out, x);

        Tensor ffn_input = ffn_norm.forward(attn_out);
        Tensor ffn_out = ffn.forward(ffn_input);
        ffn_out = AutogradEngine::add_op(ffn_out, attn_out);
        return ffn_out;
    }
}

class KimiDeltaAttention {
public:
    KimiDeltaAttention(int hidden_size, int num_heads, int head_dim);
    Tensor forward(const Tensor& x, const Tensor& positions);
private:
    Linear q_proj_, k_proj_, v_proj_, o_proj_;
    Linear delta_q_, delta_k_;
    int hidden_size_, num_heads_, head_dim_;
    float delta_ratio_; 
    
    Tensor linear_attention(const Tensor& Q, const Tensor& K, const Tensor& V);
    Tensor delta_correction(const Tensor& Q, const Tensor& K, const Tensor& V, int top_k = 64);
};

KimiDeltaAttention::KimiDeltaAttention(int hidden_size, int num_heads, int head_dim)
    : q_proj_(hidden_size, num_heads * head_dim),
      k_proj_(hidden_size, num_heads * head_dim),
      v_proj_(hidden_size, num_heads * head_dim),
      o_proj_(num_heads * head_dim, hidden_size),
      delta_q_(hidden_size, num_heads * head_dim),
      delta_k_(hidden_size, num_heads * head_dim),
      hidden_size_(hidden_size), num_heads_(num_heads), head_dim_(head_dim), delta_ratio_(0.1f) {}

Tensor KimiDeltaAttention::linear_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
    int64_t B = Q.dim(0), H = Q.dim(1), S = Q.dim(2), D = Q.dim(3);
    Tensor KV(Shape{B, H, D, D}, DType::F32);
    KV.zero_();
    
    const float* qd = Q.data<float>();
    const float* kd = K.data<float>();
    const float* vd = V.data<float>();
    float* kvd = KV.data<float>();
    
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                for (int64_t d1 = 0; d1 < D; d1++) {
                    float k_val = kd[((b * H + h) * S + s) * D + d1];
                    for (int64_t d2 = 0; d2 < D; d2++) {
                        float v_val = vd[((b * H + h) * S + s) * D + d2];
                        kvd[((b * H + h) * D + d1) * D + d2] += k_val * v_val;
                    }
                }
            }
        }
    }
    
    Tensor out(Shape{B, H, S, D}, DType::F32);
    out.zero_();
    float* od = out.data<float>();
    
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                for (int64_t d2 = 0; d2 < D; d2++) {
                    float sum = 0.0f;
                    for (int64_t d1 = 0; d1 < D; d1++) {
                        sum += qd[((b * H + h) * S + s) * D + d1] * kvd[((b * H + h) * D + d1) * D + d2];
                    }
                    od[((b * H + h) * S + s) * D + d2] = sum;
                }
            }
        }
    }
    return out;
}

Tensor KimiDeltaAttention::delta_correction(const Tensor& Q, const Tensor& K, const Tensor& V, int top_k) {
    int64_t B = Q.dim(0), H = Q.dim(1), S = Q.dim(2), D = Q.dim(3);
    Tensor out(Shape{B, H, S, D}, DType::F32);
    out.zero_();
    if (S == 0) return out;
    
    int actual_top_k = std::min((int)S, top_k);
    const float* qd = Q.data<float>();
    const float* kd = K.data<float>();
    const float* vd = V.data<float>();
    float* od = out.data<float>();
    float scale = 1.0f / std::sqrt((float)D);
    
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                std::vector<std::pair<float, int64_t>> scores;
                for (int64_t t = 0; t < S; t++) {
                    float score = 0.0f;
                    for (int64_t d = 0; d < D; d++) {
                        score += qd[((b * H + h) * S + s) * D + d] * kd[((b * H + h) * S + t) * D + d];
                    }
                    scores.push_back({score * scale, t});
                }
                
                std::partial_sort(scores.begin(), scores.begin() + actual_top_k, scores.end(),
                                  [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
                                      return a.first > b.first;
                                  });
                
                float max_score = scores[0].first;
                float sum_exp = 0.0f;
                std::vector<float> exps(actual_top_k);
                for (int k = 0; k < actual_top_k; k++) {
                    exps[k] = std::exp(scores[k].first - max_score);
                    sum_exp += exps[k];
                }
                
                for (int k = 0; k < actual_top_k; k++) {
                    float weight = exps[k] / sum_exp;
                    int64_t t = scores[k].second;
                    for (int64_t d = 0; d < D; d++) {
                        od[((b * H + h) * S + s) * D + d] += weight * vd[((b * H + h) * S + t) * D + d];
                    }
                }
            }
        }
    }
    return out;
}

Tensor KimiDeltaAttention::forward(const Tensor& x, const Tensor& positions) {
    int64_t B = x.dim(0);
    int64_t S = x.dim(1);
    
    Tensor q = q_proj_.forward(x);
    Tensor k = k_proj_.forward(x);
    Tensor v = v_proj_.forward(x);
    
    Tensor q_reshaped = q.reshape(Shape{B, S, num_heads_, head_dim_}).transpose(1, 2);
    Tensor k_reshaped = k.reshape(Shape{B, S, num_heads_, head_dim_}).transpose(1, 2);
    Tensor v_reshaped = v.reshape(Shape{B, S, num_heads_, head_dim_}).transpose(1, 2);
    
    Tensor linear_out = linear_attention(q_reshaped, k_reshaped, v_reshaped);
    
    Tensor dq = delta_q_.forward(x);
    Tensor dk = delta_k_.forward(x);
    Tensor dq_reshaped = dq.reshape(Shape{B, S, num_heads_, head_dim_}).transpose(1, 2);
    Tensor dk_reshaped = dk.reshape(Shape{B, S, num_heads_, head_dim_}).transpose(1, 2);
    
    Tensor delta_out = delta_correction(dq_reshaped, dk_reshaped, v_reshaped, 64);
    
    Tensor combined(Shape{B, num_heads_, S, head_dim_}, DType::F32);
    const float* l_d = linear_out.data<float>();
    const float* d_d = delta_out.data<float>();
    float* c_d = combined.data<float>();
    int64_t total = combined.numel();
    for(int64_t i = 0; i < total; i++) {
        c_d[i] = l_d[i] + delta_ratio_ * d_d[i];
    }
    
    Tensor flat = combined.transpose(1, 2).reshape(Shape{B * S, num_heads_ * head_dim_});
    Tensor o_out = o_proj_.forward(flat);
    return o_out.reshape(Shape{B, S, hidden_size_});
}

class AttentionResidual {
public:
    AttentionResidual(int hidden_size);
    Tensor forward(const Tensor& current_output, const Tensor& earlier_layer_output);
private:
    Linear gate_proj_;
    int hidden_size_;
};

AttentionResidual::AttentionResidual(int hidden_size) 
    : gate_proj_(hidden_size * 2, hidden_size), hidden_size_(hidden_size) {}

Tensor AttentionResidual::forward(const Tensor& current_output, const Tensor& earlier_layer_output) {
    int64_t B = current_output.dim(0);
    int64_t S = current_output.dim(1);
    int64_t D = hidden_size_;
    
    Tensor concat(Shape{B, S, D * 2}, DType::F32);
    const float* cd = current_output.data<float>();
    const float* ed = earlier_layer_output.data<float>();
    float* ccd = concat.data<float>();
    
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t d = 0; d < D; d++) {
                ccd[(b * S + s) * (D * 2) + d] = cd[(b * S + s) * D + d];
                ccd[(b * S + s) * (D * 2) + D + d] = ed[(b * S + s) * D + d];
            }
        }
    }
    
    Tensor gate = gate_proj_.forward(concat.reshape(Shape{B * S, D * 2}));
    Tensor out(Shape{B, S, D}, DType::F32);
    const float* gd = gate.data<float>();
    float* od = out.data<float>();
    
    for (int64_t i = 0; i < B * S * D; i++) {
        float g = 1.0f / (1.0f + std::exp(-gd[i]));
        od[i] = cd[i] * (1.0f - g) + ed[i] * g;
    }
    
    return out;
}

// Native FP8 (E4M3 / E5M2) Types and Quantized MatMul
struct FP8_E4M3 {
    uint8_t val;
    static float to_float(uint8_t v) {
        if (v == 0) return 0.0f;
        int sign = (v & 0x80) ? -1 : 1;
        int exp = (v & 0x78) >> 3;
        int mant = v & 0x07;
        if (exp == 0) return sign * std::pow(2.0f, -6) * (mant / 8.0f);
        if (exp == 15) return std::nanf(""); // NaN for E4M3
        return sign * std::pow(2.0f, exp - 7) * (1.0f + mant / 8.0f);
    }
    static uint8_t from_float(float f) {
        if (f == 0.0f) return 0;
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        int sign = (bits >> 31) & 1;
        int exp = ((bits >> 23) & 0xFF) - 127;
        int mant = bits & 0x7FFFFF;
        if (exp < -9) return sign << 7;
        if (exp > 8) return (sign << 7) | 0x7E;
        int target_exp = exp + 7;
        int target_mant = mant >> 20;
        return (sign << 7) | (target_exp << 3) | target_mant;
    }
};

struct FP8_E5M2 {
    uint8_t val;
    static float to_float(uint8_t v) {
        if ((v & 0x7F) == 0) return 0.0f;
        int sign = (v & 0x80) ? -1 : 1;
        int exp = (v & 0x7C) >> 2;
        int mant = v & 0x03;
        if (exp == 0) return sign * std::pow(2.0f, -14) * (mant / 4.0f);
        if (exp == 31) return (mant == 0) ? (sign * INFINITY) : std::nanf("");
        return sign * std::pow(2.0f, exp - 15) * (1.0f + mant / 4.0f);
    }
    static uint8_t from_float(float f) {
        if (f == 0.0f) return 0;
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        int sign = (bits >> 31) & 1;
        int exp = ((bits >> 23) & 0xFF) - 127;
        int mant = bits & 0x7FFFFF;
        if (exp < -16) return sign << 7;
        if (exp > 15) return (sign << 7) | 0x7C;
        int target_exp = exp + 15;
        int target_mant = mant >> 21;
        return (sign << 7) | (target_exp << 2) | target_mant;
    }
};

Tensor fp8_matmul(const Tensor& A, const Tensor& B, bool use_e4m3 = true) {
    int64_t B_batch = A.dim(0);
    int64_t M = A.dim(1);
    int64_t K = A.dim(2);
    int64_t N = B.dim(1);
    Tensor C(Shape{B_batch, M, N}, DType::F32);
    C.zero_();
    
    const float* a_d = A.data<float>();
    const float* b_d = B.data<float>();
    float* c_d = C.data<float>();
    
    for (int64_t b = 0; b < B_batch; b++) {
        for (int64_t m = 0; m < M; m++) {
            for (int64_t k = 0; k < K; k++) {
                float a_val = a_d[(b * M + m) * K + k];
                float a_q = use_e4m3 ? FP8_E4M3::to_float(FP8_E4M3::from_float(a_val)) : FP8_E5M2::to_float(FP8_E5M2::from_float(a_val));
                for (int64_t n = 0; n < N; n++) {
                    float b_val = b_d[k * N + n]; // B is K x N
                    float b_q = use_e4m3 ? FP8_E4M3::to_float(FP8_E4M3::from_float(b_val)) : FP8_E5M2::to_float(FP8_E5M2::from_float(b_val));
                    c_d[(b * M + m) * N + n] += a_q * b_q;
                }
            }
        }
    }
    return C;
}

class MultiHeadLatentAttention {
public:
    MultiHeadLatentAttention(int hidden_size, int num_heads, int rope_dim, int q_lora_rank, int kv_lora_rank);
    Tensor forward(const Tensor& x, const Tensor& positions);
private:
    Linear w_dq, w_uq, w_dkv, w_uk, w_uv, o_proj;
    Linear q_pe_proj, k_pe_proj;
    int hidden_size_, num_heads_, rope_dim_, head_dim_;
};

MultiHeadLatentAttention::MultiHeadLatentAttention(int hidden_size, int num_heads, int rope_dim, int q_lora_rank, int kv_lora_rank)
    : w_dq(hidden_size, q_lora_rank), w_uq(q_lora_rank, num_heads * (hidden_size / num_heads)),
      w_dkv(hidden_size, kv_lora_rank), w_uk(kv_lora_rank, num_heads * (hidden_size / num_heads)),
      w_uv(kv_lora_rank, num_heads * (hidden_size / num_heads)), o_proj(num_heads * (hidden_size / num_heads), hidden_size),
      q_pe_proj(q_lora_rank, rope_dim), k_pe_proj(kv_lora_rank, rope_dim),
      hidden_size_(hidden_size), num_heads_(num_heads), rope_dim_(rope_dim), head_dim_(hidden_size / num_heads) {}

Tensor MultiHeadLatentAttention::forward(const Tensor& x, const Tensor& positions) {
    int64_t B = x.dim(0);
    int64_t S = x.dim(1);
    
    // Q Compression
    Tensor c_q = w_dq.forward(x); // B x S x q_lora_rank
    Tensor q_c = w_uq.forward(c_q).reshape(Shape{B, S, num_heads_, head_dim_});
    Tensor q_pe = q_pe_proj.forward(c_q);
    
    // KV Compression
    Tensor c_kv = w_dkv.forward(x); // B x S x kv_lora_rank
    Tensor k_c = w_uk.forward(c_kv).reshape(Shape{B, S, num_heads_, head_dim_});
    Tensor v_c = w_uv.forward(c_kv).reshape(Shape{B, S, num_heads_, head_dim_});
    Tensor k_pe = k_pe_proj.forward(c_kv);
    
    // Attention calculation (Simplified for Latent space)
    Tensor q = q_c.transpose(1, 2); // B, H, S, D
    Tensor k = k_c.transpose(1, 2);
    Tensor v = v_c.transpose(1, 2);
    
    Tensor out(Shape{B, num_heads_, S, head_dim_}, DType::F32);
    out.zero_();
    const float* qd = q.data<float>();
    const float* kd = k.data<float>();
    const float* vd = v.data<float>();
    float* od = out.data<float>();
    float scale = 1.0f / std::sqrt((float)head_dim_);
    
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < num_heads_; h++) {
            for (int64_t s = 0; s < S; s++) {
                std::vector<float> scores(S, 0.0f);
                float max_score = -1e9f;
                for (int64_t t = 0; t <= s; t++) {
                    float score = 0;
                    for (int64_t d = 0; d < head_dim_; d++) {
                        score += qd[((b * num_heads_ + h) * S + s) * head_dim_ + d] * kd[((b * num_heads_ + h) * S + t) * head_dim_ + d];
                    }
                    score *= scale;
                    scores[t] = score;
                    if (score > max_score) max_score = score;
                }
                float sum_exp = 0;
                for (int64_t t = 0; t <= s; t++) {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                for (int64_t t = 0; t <= s; t++) {
                    float w = scores[t] / sum_exp;
                    for (int64_t d = 0; d < head_dim_; d++) {
                        od[((b * num_heads_ + h) * S + s) * head_dim_ + d] += w * vd[((b * num_heads_ + h) * S + t) * head_dim_ + d];
                    }
                }
            }
        }
    }
    
    Tensor flat_out = out.transpose(1, 2).reshape(Shape{B * S, num_heads_ * head_dim_});
    return o_proj.forward(flat_out).reshape(Shape{B, S, hidden_size_});
}

class MultiTokenPredictionHead {
public:
    MultiTokenPredictionHead(int hidden_size, int vocab_size, int num_heads);
    std::vector<Tensor> forward(const Tensor& hidden_states);
private:
    std::vector<Linear> heads_;
    int num_heads_;
};

MultiTokenPredictionHead::MultiTokenPredictionHead(int hidden_size, int vocab_size, int num_heads) : num_heads_(num_heads) {
    for (int i = 0; i < num_heads; i++) {
        heads_.emplace_back(hidden_size, vocab_size);
    }
}

std::vector<Tensor> MultiTokenPredictionHead::forward(const Tensor& hidden_states) {
    std::vector<Tensor> predictions;
    for (int i = 0; i < num_heads_; i++) {
        predictions.push_back(heads_[i].forward(hidden_states));
    }
    return predictions;
}

} // namespace quant
