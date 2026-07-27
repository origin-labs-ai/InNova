#include "oil/transformer.h"
#include "oil/math.h"
#include "oil/autograd.h"
#include "oil/random.h"
#include "oil/simd_math.h"
#include "oil/flash_attention.h"
#include <cmath>
#include <cstdio>
#include <cstring>

#if defined(OIL_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace oil {

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
#if defined(OIL_AVX2) || defined(__AVX2__)
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
#if defined(OIL_AVX2) || defined(__AVX2__)
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
      head_dim(cfg.head_dim),
      rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
{
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
        // Inference path: transpose {B,S,H,D} -> {B,H,S,D}, then RoPE + KV cache + attention
        Tensor q_t = q_reshaped.transpose(1, 2);  // {B, H, S, D}
        Tensor k_t = k_reshaped.transpose(1, 2);  // {B, KV_H, S, D}
        Tensor v_t = v_reshaped.transpose(1, 2);  // {B, KV_H, S, D}
        
        int64_t seq_start = cache.context_len();
        rope.apply(q_t, seq_start, S);
        rope.apply(k_t, seq_start, S);
        cache.append(layer_idx, k_t, v_t);
        auto [k_full, v_full] = cache.get_all(layer_idx);
        int64_t S_full = k_full.shape().dims[2];
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
                               k_full.data<float>() + src_off, head_dim * sizeof(float));
                        memcpy(v_expanded.data<float>() + dst_off,
                               v_full.data<float>() + src_off, head_dim * sizeof(float));
                    }
                }
            }
        } else {
            k_expanded = k_full;
            v_expanded = v_full;
        }

        // Use FlashAttention for long sequences (memory O(n) vs O(n²))
        if (S_full > 64) {
            Tensor causal_mask(Shape{1, 1, S, S_full}, DType::F32);
            float* md = causal_mask.data<float>();
            for (int64_t s = 0; s < S; s++) {
                for (int64_t t = 0; t < S_full; t++) {
                    md[s * S_full + t] = (t > s + seq_start - S) ? -INFINITY : 0.0f;
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
                            if (t > s + seq_start - S)
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
    
    if (activation == Activation::SiLU) {
        gate = AutogradEngine::silu_op(gate);
    } else if (activation == Activation::GELU) {
        math::gelu(gate, gate);
    } else {
        math::relu(gate, gate);
    }
    
    Tensor hidden = AutogradEngine::mul_op(gate, up);
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

} // namespace oil
