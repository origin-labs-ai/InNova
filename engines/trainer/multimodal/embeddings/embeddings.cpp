#include "embeddings.h"
#include "oil/math.h"
#include "oil/random.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace oil {
namespace multimodal {

static void linear_forward(const Tensor& w, const Tensor& b, const Tensor& x, Tensor& out) {
    int64_t N = x.dim(0), D = w.dim(0);
    math::gemm(1.0f, x, w, 0.0f, out);
    if (b.numel() > 0) {
        float* od = out.data<float>();
        const float* bd = b.data<float>();
        for (int64_t i = 0; i < N; i++)
            for (int64_t j = 0; j < D; j++)
                od[i * D + j] += bd[j];
    }
}

EmbeddingEncoder::EmbeddingEncoder(const EmbeddingConfig& cfg)
    : input_dim(cfg.input_dim), hidden_dim(cfg.hidden_dim) {
    RNG rng(42);
    input_proj_w = Tensor::zeros({cfg.input_dim, cfg.hidden_dim});
    float* w1 = input_proj_w.data<float>();
    for (int64_t i = 0; i < input_proj_w.numel(); i++) w1[i] = rng.normal() * 0.02f;
    input_proj_b = Tensor::zeros({cfg.hidden_dim});
    hidden_proj_w = Tensor::zeros({cfg.hidden_dim, cfg.hidden_dim});
    float* w2 = hidden_proj_w.data<float>();
    for (int64_t i = 0; i < hidden_proj_w.numel(); i++) w2[i] = rng.normal() * 0.02f;
    hidden_proj_b = Tensor::zeros({cfg.hidden_dim});
}

Tensor EmbeddingEncoder::mean_pool(const Tensor& token_embs, const Tensor& mask) const {
    int64_t B = token_embs.dim(0), S = token_embs.dim(1), D = token_embs.dim(2);
    Tensor pooled = Tensor::zeros({B, D});
    float* pd = pooled.data<float>();
    const float* td = token_embs.data<float>();
    const float* md = mask.data<float>();
    for (int64_t b = 0; b < B; b++) {
        float sum_mask = 0;
        for (int64_t s = 0; s < S; s++) sum_mask += md[b * S + s];
        if (sum_mask < 0.5f) sum_mask = 1.0f;
        float inv_sum = 1.0f / sum_mask;
        for (int64_t s = 0; s < S; s++) {
            float w = md[b * S + s] * inv_sum;
            for (int64_t d = 0; d < D; d++)
                pd[b * D + d] += w * td[b * S * D + s * D + d];
        }
    }
    return pooled;
}

Tensor EmbeddingEncoder::cls_pool(const Tensor& token_embs, const Tensor& cls_weight) const {
    int64_t B = token_embs.dim(0), S = token_embs.dim(1), D = token_embs.dim(2);
    Tensor pooled({B, D});
    pooled.zero_();
    float* pd = pooled.data<float>();
    const float* td = token_embs.data<float>();
    const float* cw = cls_weight.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++) {
            float attn = 0;
            for (int64_t d = 0; d < D; d++)
                attn += td[b * S * D + s * D + d] * cw[d];
            attn = std::exp(attn);
            for (int64_t d = 0; d < D; d++)
                pd[b * D + d] += attn * td[b * S * D + s * D + d];
        }
    return pooled;
}

Tensor EmbeddingEncoder::max_pool(const Tensor& token_embs, const Tensor& mask) const {
    int64_t B = token_embs.dim(0), S = token_embs.dim(1), D = token_embs.dim(2);
    Tensor pooled({B, D});
    const float* td = token_embs.data<float>();
    const float* md = mask.data<float>();
    float* pd = pooled.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t d = 0; d < D; d++) {
            float maxv = -1e30f;
            for (int64_t s = 0; s < S; s++) {
                if (md[b * S + s] > 0.5f)
                    maxv = std::max(maxv, td[b * S * D + s * D + d]);
            }
            pd[b * D + d] = (maxv > -1e29f) ? maxv : 0.0f;
        }
    return pooled;
}

Tensor EmbeddingEncoder::encode_text(const Tensor& text_embeddings, const Tensor& attention_mask) const {
    int64_t B = text_embeddings.dim(0), S = text_embeddings.dim(1), D = text_embeddings.dim(2);
    Tensor flat = text_embeddings.reshape({B * S, D});
    Tensor projected({B * S, hidden_dim});
    linear_forward(input_proj_w, input_proj_b, flat, projected);
    Tensor h = projected.reshape({B, S, hidden_dim});
    Tensor pooled = mean_pool(h, attention_mask);
    Tensor out({B, hidden_dim});
    linear_forward(hidden_proj_w, hidden_proj_b, pooled, out);
    return out;
}

ContrastiveHead::ContrastiveHead(int64_t input_dim, int64_t emb_dim)
    : embedding_dim(emb_dim) {
    RNG rng(42);
    projection_w = Tensor::zeros({input_dim, emb_dim});
    float* w = projection_w.data<float>();
    for (int64_t i = 0; i < projection_w.numel(); i++) w[i] = rng.normal() * 0.02f;
    projection_b = Tensor::zeros({emb_dim});
}

void ContrastiveHead::l2_normalize(Tensor& x) const {
    int64_t B = x.dim(0), D = x.dim(1);
    float* xd = x.data<float>();
    for (int64_t b = 0; b < B; b++) {
        float norm = 0;
        for (int64_t d = 0; d < D; d++)
            norm += xd[b * D + d] * xd[b * D + d];
        norm = std::sqrt(norm + 1e-10f);
        float inv_norm = 1.0f / norm;
        for (int64_t d = 0; d < D; d++)
            xd[b * D + d] *= inv_norm;
    }
}

Tensor ContrastiveHead::forward(const Tensor& x) const {
    int64_t N = x.dim(0);
    Tensor out({N, embedding_dim});
    linear_forward(projection_w, projection_b, x, out);
    l2_normalize(out);
    return out;
}

Tensor ContrastiveHead::compute_similarity(const Tensor& emb1, const Tensor& emb2) const {
    int64_t B1 = emb1.dim(0), B2 = emb2.dim(0), D = emb1.dim(1);
    Tensor sim = Tensor::zeros({B1, B2});
    float* sd = sim.data<float>();
    const float* e1 = emb1.data<float>();
    const float* e2 = emb2.data<float>();
    for (int64_t i = 0; i < B1; i++)
        for (int64_t j = 0; j < B2; j++) {
            float dot = 0;
            for (int64_t d = 0; d < D; d++)
                dot += e1[i * D + d] * e2[j * D + d];
            sd[i * B2 + j] = dot;
        }
    return sim;
}

// NTXentLoss
NTXentLoss::NTXentLoss(float temperature) : temperature(temperature) {}

Tensor NTXentLoss::compute_logits(const Tensor& emb1, const Tensor& emb2) const {
    int64_t B1 = emb1.dim(0), B2 = emb2.dim(0), D = emb1.dim(1);
    Tensor logits({B1, B2});
    float* ld = logits.data<float>();
    const float* e1 = emb1.data<float>();
    const float* e2 = emb2.data<float>();
    for (int64_t i = 0; i < B1; i++)
        for (int64_t j = 0; j < B2; j++) {
            float dot = 0;
            for (int64_t d = 0; d < D; d++)
                dot += e1[i * D + d] * e2[j * D + d];
            ld[i * B2 + j] = dot / temperature;
        }
    return logits;
}

float NTXentLoss::forward(const Tensor& emb1, const Tensor& emb2) const {
    int64_t B = emb1.dim(0);
    OIL_CHECK(B > 0, "NTXentLoss: batch must be > 0");
    OIL_CHECK(emb1.dim(0) == emb2.dim(0) && emb1.dim(1) == emb2.dim(1),
              "NTXentLoss: shape mismatch");
    Tensor logits = compute_logits(emb1, emb2);
    const float* ld = logits.data<float>();
    float loss = 0.0f;
    for (int64_t i = 0; i < B; i++) {
        float max_row = ld[i * B];
        for (int64_t j = 1; j < B; j++)
            if (ld[i * B + j] > max_row) max_row = ld[i * B + j];
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < B; j++) {
            if (j == i) continue;
            sum_exp += std::exp(ld[i * B + j] - max_row);
        }
        float log_prob = ld[i * B + i] - max_row - std::log(sum_exp + 1e-10f);
        loss -= log_prob;
    }
    return loss / (float)B;
}

// DualEncoder
DualEncoder::DualEncoder(const EmbeddingConfig& cfg)
    : query_encoder(cfg), doc_encoder(cfg),
      query_proj(cfg.hidden_dim, cfg.embedding_dim),
      doc_proj(cfg.hidden_dim, cfg.embedding_dim),
      embedding_dim(cfg.embedding_dim) {}

Tensor DualEncoder::encode_query(const Tensor& query_emb, const Tensor& query_mask) const {
    Tensor h = query_encoder.encode_text(query_emb, query_mask);
    return query_proj.forward(h);
}

Tensor DualEncoder::encode_doc(const Tensor& doc_emb, const Tensor& doc_mask) const {
    Tensor h = doc_encoder.encode_text(doc_emb, doc_mask);
    return doc_proj.forward(h);
}

float DualEncoder::compute_contrastive_loss(const Tensor& query_emb, const Tensor& doc_emb,
                                              const Tensor& query_mask, const Tensor& doc_mask) const {
    Tensor q = encode_query(query_emb, query_mask);
    Tensor d = encode_doc(doc_emb, doc_mask);
    return NTXentLoss(0.07f).forward(q, d);
}

Tensor DualEncoder::score_candidates(const Tensor& query_emb, const std::vector<Tensor>& doc_embs) const {
    int64_t N = (int64_t)doc_embs.size();
    Tensor scores({N});
    float* sd = scores.data<float>();
    const float* qd = query_emb.data<float>();
    int64_t D = query_emb.dim(1);
    for (int64_t i = 0; i < N; i++) {
        const float* dd = doc_embs[(size_t)i].data<float>();
        float dot = 0;
        for (int64_t d = 0; d < D; d++)
            dot += qd[d] * dd[d];
        sd[i] = dot;
    }
    return scores;
}

// CrossEncoder
CrossEncoder::CrossEncoder(int64_t input_dim, int64_t hidden_dim)
    : input_proj(input_dim * 2, hidden_dim),
      hidden_proj(hidden_dim, hidden_dim),
      score_head(hidden_dim, 1),
      hidden_size(hidden_dim) {}

Tensor CrossEncoder::forward(const Tensor& query_emb, const Tensor& doc_emb,
                               const Tensor& query_mask, const Tensor& doc_mask) const {
    int64_t B = query_emb.dim(0), S = query_emb.dim(1), D = query_emb.dim(2);
    OIL_CHECK(doc_emb.dim(0) == B && doc_emb.dim(1) == S && doc_emb.dim(2) == D,
              "CrossEncoder: shape mismatch");
    Tensor combined({B, S, D * 2});
    float* cd = combined.data<float>();
    const float* qd = query_emb.data<float>();
    const float* dd = doc_emb.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++) {
            std::memcpy(cd + (b * S + s) * D * 2, qd + (b * S + s) * D, (size_t)D * sizeof(float));
            std::memcpy(cd + (b * S + s) * D * 2 + D, dd + (b * S + s) * D, (size_t)D * sizeof(float));
        }
    Tensor flat = combined.reshape({B * S, D * 2});
    Tensor h = input_proj.forward(flat);
    float* hd = h.data<float>();
    for (int64_t i = 0; i < B * S * hidden_size; i++)
        hd[i] = std::tanh(hd[i]);
    h = hidden_proj.forward(h);
    float* hd2 = h.data<float>();
    for (int64_t i = 0; i < B * S * hidden_size; i++)
        hd2[i] = std::tanh(hd2[i]);
    Tensor logits_flat = score_head.forward(h);
    Tensor mask_combined = Tensor::zeros({B, S});
    const float* qm = query_mask.data<float>();
    const float* dm = doc_mask.data<float>();
    float* mcd = mask_combined.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++)
            mcd[b * S + s] = (qm[b * S + s] > 0.5f && dm[b * S + s] > 0.5f) ? 1.0f : 0.0f;
    Tensor scores({B});
    scores.zero_();
    float* sd = scores.data<float>();
    const float* lfd = logits_flat.data<float>();
    for (int64_t b = 0; b < B; b++) {
        float sum_w = 0;
        for (int64_t s = 0; s < S; s++) {
            float w = mcd[b * S + s];
            sd[b] += w * lfd[(b * S + s)];
            sum_w += w;
        }
        if (sum_w > 0.5f) sd[b] /= sum_w;
    }
    return scores;
}

float CrossEncoder::score_pair(const Tensor& query_emb, const Tensor& doc_emb) const {
    int64_t B = 1;
    int64_t S = query_emb.dim(0);
    int64_t D = query_emb.dim(1);
    Tensor q_batch = query_emb.reshape({B, S, D});
    Tensor d_batch = doc_emb.reshape({B, S, D});
    Tensor q_mask({B, S});
    q_mask.fill(1.0f);
    Tensor d_mask({B, S});
    d_mask.fill(1.0f);
    Tensor scores = forward(q_batch, d_batch, q_mask, d_mask);
    return scores.data<float>()[0];
}

Tensor CrossEncoder::score_batch(const Tensor& query_emb, const std::vector<Tensor>& doc_embs,
                                   const Tensor& query_mask) const {
    int64_t N = (int64_t)doc_embs.size();
    Tensor scores({N});
    float* sd = scores.data<float>();
    int64_t S = query_emb.dim(0);
    int64_t D = query_emb.dim(1);
    Tensor q_batch = query_emb.reshape({1, S, D});
    for (int64_t i = 0; i < N; i++) {
        Tensor d_batch = doc_embs[(size_t)i].reshape({1, S, D});
        Tensor d_mask({1, S});
        d_mask.fill(1.0f);
        Tensor s = forward(q_batch, d_batch, query_mask.reshape({1, S}), d_mask);
        sd[i] = s.data<float>()[0];
    }
    return scores;
}

} // namespace multimodal
} // namespace oil
