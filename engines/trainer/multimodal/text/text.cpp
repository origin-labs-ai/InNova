#include "text.h"
#include "oil/math.h"
#include "oil/random.h"
#include "oil/kv_cache.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace oil {
namespace multimodal {

// SinusoidalPositionalEncoding
SinusoidalPositionalEncoding::SinusoidalPositionalEncoding(int64_t max_len, int64_t dim)
    : max_len(max_len), dim(dim) {
    pe = Tensor::zeros(Shape{max_len, dim});
    float* pd = pe.data<float>();
    for (int64_t pos = 0; pos < max_len; pos++) {
        for (int64_t i = 0; i < dim / 2; i++) {
            float inv_freq = 1.0f / std::pow(10000.0f, (float)(2 * i) / (float)dim);
            float val = (float)pos * inv_freq;
            pd[pos * dim + i] = std::sin(val);
            pd[pos * dim + dim / 2 + i] = std::cos(val);
        }
    }
}

Tensor SinusoidalPositionalEncoding::forward(int64_t seq_len) const {
    if (seq_len <= max_len)
        return pe.slice(0, 0, seq_len).clone();
    return pe.clone();
}

// LearnablePositionalEncoding
LearnablePositionalEncoding::LearnablePositionalEncoding(int64_t max_len, int64_t dim)
    : max_len(max_len), dim(dim) {
    RNG rng(42);
    weight = Tensor::zeros(Shape{max_len, dim});
    float* wd = weight.data<float>();
    for (int64_t i = 0; i < weight.numel(); i++)
        wd[i] = rng.normal() * 0.02f;
}

Tensor LearnablePositionalEncoding::forward(int64_t seq_len) const {
    if (seq_len <= max_len)
        return weight.slice(0, 0, seq_len).clone();
    return weight.clone();
}

// SequenceClassifier
SequenceClassifier::SequenceClassifier(int64_t hidden, int64_t n_classes, float dropout)
    : dense(hidden, hidden), classifier(hidden, n_classes),
      hidden_size(hidden), num_classes(n_classes), dropout_ratio(dropout) {}

Tensor SequenceClassifier::forward(const Tensor& pooled) const {
    int64_t B = pooled.dim(0), D = pooled.dim(1);
    Tensor h = dense.forward(pooled);
    float* hd = h.data<float>();
    for (int64_t i = 0; i < B * D; i++)
        hd[i] = std::tanh(hd[i]);
    Tensor out = classifier.forward(h);
    return out;
}

Tensor SequenceClassifier::predict_proba(const Tensor& pooled) const {
    Tensor logits = forward(pooled);
    int64_t B = logits.dim(0), C = logits.dim(1);
    Tensor proba({B, C});
    const float* ld = logits.data<float>();
    float* pd = proba.data<float>();
    for (int64_t b = 0; b < B; b++) {
        float maxv = ld[b * C];
        for (int64_t c = 1; c < C; c++)
            if (ld[b * C + c] > maxv) maxv = ld[b * C + c];
        float sum_exp = 0.0f;
        for (int64_t c = 0; c < C; c++) {
            pd[b * C + c] = std::exp(ld[b * C + c] - maxv);
            sum_exp += pd[b * C + c];
        }
        if (sum_exp > 1e-10f) {
            float inv = 1.0f / sum_exp;
            for (int64_t c = 0; c < C; c++)
                pd[b * C + c] *= inv;
        }
    }
    return proba;
}

int64_t SequenceClassifier::predict_class(const Tensor& pooled) const {
    Tensor proba = predict_proba(pooled);
    const float* pd = proba.data<float>();
    int64_t C = proba.dim(1);
    int64_t best = 0;
    float maxv = pd[0];
    for (int64_t c = 1; c < C; c++)
        if (pd[c] > maxv) { maxv = pd[c]; best = c; }
    return best;
}

size_t SequenceClassifier::param_count() const {
    return (size_t)(dense.weight.numel() + dense.bias.numel() +
                    classifier.weight.numel() + classifier.bias.numel());
}

// BPEEncoder
BPEEncoder::BPEEncoder(int64_t max_seq_len) : max_seq_len(max_seq_len) {}

std::vector<int> BPEEncoder::encode_text(const std::string& text) {
    return tokenizer.encode(text);
}

Tensor BPEEncoder::tokenize_to_tensor(const std::string& text) {
    std::vector<int> ids = tokenizer.encode(text);
    if ((int64_t)ids.size() > max_seq_len - 1)
        ids.resize((size_t)max_seq_len - 1);
    int64_t S = (int64_t)ids.size() + 1;
    Tensor tokens({1, S});
    float* td = tokens.data<float>();
    td[0] = (float)tokenizer.bos_id();
    for (int64_t i = 0; i < (int64_t)ids.size(); i++)
        td[i + 1] = (float)ids[(size_t)i];
    return tokens;
}

Tensor BPEEncoder::compute_attention_mask(const Tensor& token_ids) const {
    int64_t B = token_ids.dim(0), S = token_ids.dim(1);
    Tensor mask({B, S});
    mask.fill(0.0f);
    float* md = mask.data<float>();
    const float* td = token_ids.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++)
            md[b * S + s] = (td[b * S + s] != 0.0f || s == 0) ? 1.0f : 0.0f;
    return mask;
}

std::string BPEEncoder::decode_tokens(const Tensor& token_ids) {
    int64_t S = token_ids.numel();
    const float* td = token_ids.data<float>();
    std::vector<int> ids((size_t)S);
    for (int64_t i = 0; i < S; i++)
        ids[(size_t)i] = (int)td[i];
    return tokenizer.decode(ids);
}

// Pooling functions
Tensor apply_cls_pool(const Tensor& token_embs, int64_t cls_pos) {
    int64_t B = token_embs.dim(0), D = token_embs.dim(2);
    Tensor pooled({B, D});
    const float* td = token_embs.data<float>();
    float* pd = pooled.data<float>();
    for (int64_t b = 0; b < B; b++)
        std::memcpy(pd + b * D, td + b * token_embs.dim(1) * D + cls_pos * D, D * sizeof(float));
    return pooled;
}

Tensor apply_mean_pool(const Tensor& token_embs, const Tensor& mask) {
    int64_t B = token_embs.dim(0), S = token_embs.dim(1), D = token_embs.dim(2);
    Tensor pooled({B, D});
    pooled.zero_();
    float* pd = pooled.data<float>();
    const float* td = token_embs.data<float>();
    const float* md = mask.numel() > 0 ? mask.data<float>() : nullptr;
    for (int64_t b = 0; b < B; b++) {
        float sum_mask = 0.0f;
        for (int64_t s = 0; s < S; s++) {
            float m = md ? md[b * S + s] : 1.0f;
            sum_mask += m;
            if (m > 0.5f)
                for (int64_t d = 0; d < D; d++)
                    pd[b * D + d] += td[b * S * D + s * D + d];
        }
        if (sum_mask > 0.5f) {
            float inv = 1.0f / sum_mask;
            for (int64_t d = 0; d < D; d++)
                pd[b * D + d] *= inv;
        }
    }
    return pooled;
}

Tensor apply_max_pool(const Tensor& token_embs, const Tensor& mask) {
    int64_t B = token_embs.dim(0), S = token_embs.dim(1), D = token_embs.dim(2);
    Tensor pooled({B, D});
    const float* td = token_embs.data<float>();
    const float* md = mask.numel() > 0 ? mask.data<float>() : nullptr;
    float* pd = pooled.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t d = 0; d < D; d++) {
            float maxv = -1e30f;
            for (int64_t s = 0; s < S; s++) {
                float m = md ? md[b * S + s] : 1.0f;
                if (m > 0.5f) {
                    float v = td[b * S * D + s * D + d];
                    if (v > maxv) maxv = v;
                }
            }
            pd[b * D + d] = (maxv > -1e29f) ? maxv : 0.0f;
        }
    return pooled;
}

// TextEncoder
TextEncoder::TextEncoder(const TextConfig& cfg)
    : embedding(cfg.vocab_size, cfg.hidden_size),
      pos_enc(cfg.max_seq_len, cfg.hidden_size),
      final_norm(cfg.hidden_size, cfg.norm_eps),
      hidden_size(cfg.hidden_size) {
    blocks.reserve(cfg.num_layers);
    for (int64_t i = 0; i < cfg.num_layers; i++)
        blocks.emplace_back(
            TransformerConfig{cfg.vocab_size, cfg.hidden_size, 1,
                              cfg.num_heads, cfg.head_dim, cfg.ffn_hidden,
                              cfg.norm_eps, 10000.0f, cfg.max_seq_len,
                              Activation::GELU, cfg.num_heads, false});
}

Tensor TextEncoder::encode(const Tensor& tokens) const {
    int64_t B = tokens.dim(0);
    int64_t S = tokens.dim(1);
    Tensor h = embedding.forward(tokens.reshape(Shape{B * S}));
    h = h.reshape(Shape{B, S, hidden_size});
    Tensor pos = pos_enc.forward(S);
    const float* pd = pos.data<float>();
    float* hd = h.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++)
            for (int64_t d = 0; d < hidden_size; d++)
                hd[b * S * hidden_size + s * hidden_size + d] += pd[s * hidden_size + d];
    KVCache cache;
    Tensor dummy_pos = Tensor::arange(S).reshape(Shape{B, S});
    Tensor causal_mask({S, S});
    causal_mask.fill(-1e10f);
    float* md = causal_mask.data<float>();
    for (int64_t i = 0; i < S; i++)
        for (int64_t j = 0; j <= i; j++)
            md[i * S + j] = 0.0f;
    for (const auto& block : blocks)
        h = block.forward(h, dummy_pos, causal_mask, cache, 0);
    return final_norm.forward(h);
}

Tensor TextEncoder::encode_with_pooling(const Tensor& tokens, const Tensor& mask, PoolingStrategy strategy) const {
    Tensor encoded = encode(tokens);
    if (strategy == PoolingStrategy::CLS_TOKEN)
        return apply_cls_pool(encoded, 0);
    else if (strategy == PoolingStrategy::MEAN)
        return apply_mean_pool(encoded, mask);
    else
        return apply_max_pool(encoded, mask);
}

Tensor TextEncoder::classify(const Tensor& tokens, const SequenceClassifier& classifier,
                              const Tensor& mask, PoolingStrategy strategy) const {
    Tensor pooled = encode_with_pooling(tokens, mask, strategy);
    return classifier.forward(pooled);
}

size_t TextEncoder::param_count() const {
    size_t count = (size_t)embedding.param_count();
    for (const auto& block : blocks) {
        count += (size_t)(block.attention.q_proj.param_count() +
                          block.attention.k_proj.param_count() +
                          block.attention.v_proj.param_count() +
                          block.attention.o_proj.param_count() +
                          block.ffn.gate_proj.param_count() +
                          block.ffn.up_proj.param_count() +
                          block.ffn.down_proj.param_count() +
                          block.attention_norm.weight.numel() +
                          block.ffn_norm.weight.numel());
    }
    count += (size_t)final_norm.weight.numel();
    return count;
}

// TextDecoder
TextDecoder::TextDecoder(const TextConfig& cfg)
    : embedding(cfg.vocab_size, cfg.hidden_size),
      pos_enc(cfg.max_seq_len, cfg.hidden_size),
      lm_head(cfg.hidden_size, cfg.vocab_size),
      final_norm(cfg.hidden_size, cfg.norm_eps),
      hidden_size(cfg.hidden_size),
      vocab_size(cfg.vocab_size) {
    blocks.reserve(cfg.num_layers);
    for (int64_t i = 0; i < cfg.num_layers; i++)
        blocks.emplace_back(
            TransformerConfig{cfg.vocab_size, cfg.hidden_size, 1,
                              cfg.num_heads, cfg.head_dim, cfg.ffn_hidden,
                              cfg.norm_eps, 10000.0f, cfg.max_seq_len,
                              Activation::SiLU, cfg.num_heads, false});
}

Tensor TextDecoder::decode(const Tensor& hidden_state, int64_t max_len) const {
    int64_t B = hidden_state.dim(0);
    int64_t S = hidden_state.dim(1);
    Tensor tokens = Tensor::zeros(Shape{B, max_len});
    float* td = tokens.data<float>();
    for (int64_t b = 0; b < B; b++)
        td[b * max_len] = 1;
    Tensor pos = pos_enc.forward(max_len);
    const float* pd = pos.data<float>();
    KVCache cache;
    for (int64_t t = 0; t < max_len; t++) {
        Tensor input = tokens.slice(1, t, t + 1);
        Tensor h = embedding.forward(input.reshape(Shape{B}));
        h = h.reshape(Shape{B, 1, hidden_size});
        float* hdp = h.data<float>();
        for (int64_t b = 0; b < B; b++)
            for (int64_t d = 0; d < hidden_size; d++)
                hdp[b * hidden_size + d] += pd[t * hidden_size + d];
        Tensor dummy_pos = Tensor::arange(t + 1).reshape(Shape{B, t + 1});
        Tensor causal_mask({t + 1, t + 1});
        causal_mask.fill(-1e10f);
        float* md = causal_mask.data<float>();
        for (int64_t i = 0; i <= t; i++)
            for (int64_t j = 0; j <= i; j++)
                md[i * (t + 1) + j] = 0.0f;
        for (const auto& block : blocks)
            h = block.forward(h, dummy_pos.slice(1, t, t + 1), causal_mask.slice(0, t, t + 1), cache, 0);
        Tensor norm_h = final_norm.forward(h);
        Tensor logits = lm_head.forward(norm_h.reshape(Shape{B, hidden_size}));
        const float* ld = logits.data<float>();
        if (t + 1 < max_len) {
            for (int64_t b = 0; b < B; b++) {
                int next = 0;
                float maxv = ld[b * vocab_size];
                for (int64_t v = 1; v < vocab_size; v++)
                    if (ld[b * vocab_size + v] > maxv) { maxv = ld[b * vocab_size + v]; next = (int)v; }
                td[b * max_len + t + 1] = (float)next;
            }
        }
    }
    return tokens;
}

Tensor TextDecoder::generate(const Tensor& tokens, int64_t max_new_tokens, const Tensor& mask) const {
    int64_t B = tokens.dim(0);
    int64_t S = tokens.dim(1);
    int64_t max_out = S + max_new_tokens;
    Tensor output = tokens.clone();
    float* od = output.data<float>();
    KVCache cache;
    for (int64_t t = 0; t < max_new_tokens; t++) {
        int64_t cur_len = output.dim(1);
        Tensor pos_enc_t = pos_enc.forward(cur_len);
        Tensor h = embedding.forward(output.reshape(Shape{B * cur_len}));
        h = h.reshape(Shape{B, cur_len, hidden_size});
        float* hd = h.data<float>();
        const float* pd = pos_enc_t.data<float>();
        for (int64_t b = 0; b < B; b++)
            for (int64_t s = 0; s < cur_len; s++)
                for (int64_t d = 0; d < hidden_size; d++)
                    hd[b * cur_len * hidden_size + s * hidden_size + d] += pd[s * hidden_size + d];
        Tensor causal_mask({cur_len, cur_len});
        causal_mask.fill(-1e10f);
        float* md = causal_mask.data<float>();
        for (int64_t i = 0; i < cur_len; i++)
            for (int64_t j = 0; j <= i; j++)
                md[i * cur_len + j] = 0.0f;
        Tensor positions = Tensor::arange(cur_len).reshape(Shape{B, cur_len});
        Tensor h_out = h;
        for (const auto& block : blocks)
            h_out = block.forward(h_out, positions, causal_mask, cache, 0);
        h_out = final_norm.forward(h_out);
        Tensor logits = lm_head.forward(h_out.reshape(Shape{B * cur_len, hidden_size}));
        const float* ld = logits.data<float>();
        Tensor new_out({B, cur_len + 1});
        float* nd = new_out.data<float>();
        for (int64_t b = 0; b < B; b++) {
            for (int64_t s = 0; s < cur_len; s++)
                nd[b * (cur_len + 1) + s] = output.data<float>()[b * cur_len + s];
            int next = 0;
            float maxv = ld[(b * cur_len + cur_len - 1) * vocab_size];
            for (int64_t v = 1; v < vocab_size; v++)
                if (ld[(b * cur_len + cur_len - 1) * vocab_size + v] > maxv) {
                    maxv = ld[(b * cur_len + cur_len - 1) * vocab_size + v];
                    next = (int)v;
                }
            nd[b * (cur_len + 1) + cur_len] = (float)next;
        }
        output = new_out;
    }
    return output;
}

size_t TextDecoder::param_count() const {
    size_t count = (size_t)embedding.param_count() + (size_t)lm_head.param_count();
    for (const auto& block : blocks) {
        count += (size_t)(block.attention.q_proj.param_count() +
                          block.attention.k_proj.param_count() +
                          block.attention.v_proj.param_count() +
                          block.attention.o_proj.param_count() +
                          block.ffn.gate_proj.param_count() +
                          block.ffn.up_proj.param_count() +
                          block.ffn.down_proj.param_count() +
                          block.attention_norm.weight.numel() +
                          block.ffn_norm.weight.numel());
    }
    count += (size_t)final_norm.weight.numel();
    return count;
}

} // namespace multimodal
} // namespace oil
