#pragma once
#include "quant/tensor.h"
#include "quant/math.h"
#include "quant/transformer.h"
#include <vector>

namespace quant {
namespace multimodal {

struct EmbeddingConfig {
    int64_t input_dim = 768;
    int64_t hidden_dim = 512;
    int64_t embedding_dim = 256;
    int64_t projection_dim = 128;
    float temperature = 0.07f;
};

class EmbeddingEncoder {
public:
    Tensor input_proj_w;
    Tensor input_proj_b;
    Tensor hidden_proj_w;
    Tensor hidden_proj_b;
    int64_t input_dim;
    int64_t hidden_dim;

    explicit EmbeddingEncoder(const EmbeddingConfig& cfg = EmbeddingConfig());
    Tensor encode_text(const Tensor& text_embeddings, const Tensor& attention_mask) const;
    Tensor mean_pool(const Tensor& token_embs, const Tensor& mask) const;
    Tensor cls_pool(const Tensor& token_embs, const Tensor& cls_weight) const;
    Tensor max_pool(const Tensor& token_embs, const Tensor& mask) const;
};

class ContrastiveHead {
public:
    Tensor projection_w;
    Tensor projection_b;
    int64_t embedding_dim;

    explicit ContrastiveHead(int64_t input_dim, int64_t emb_dim);
    Tensor forward(const Tensor& x) const;
    Tensor compute_similarity(const Tensor& emb1, const Tensor& emb2) const;
    void l2_normalize(Tensor& x) const;
};

class NTXentLoss {
public:
    float temperature;

    explicit NTXentLoss(float temperature = 0.07f);
    float forward(const Tensor& emb1, const Tensor& emb2) const;
    Tensor compute_logits(const Tensor& emb1, const Tensor& emb2) const;
};

class DualEncoder {
public:
    EmbeddingEncoder query_encoder;
    EmbeddingEncoder doc_encoder;
    ContrastiveHead query_proj;
    ContrastiveHead doc_proj;
    int64_t embedding_dim;

    DualEncoder(const EmbeddingConfig& cfg = EmbeddingConfig());
    Tensor encode_query(const Tensor& query_emb, const Tensor& query_mask) const;
    Tensor encode_doc(const Tensor& doc_emb, const Tensor& doc_mask) const;
    float compute_contrastive_loss(const Tensor& query_emb, const Tensor& doc_emb, const Tensor& query_mask, const Tensor& doc_mask) const;
    Tensor score_candidates(const Tensor& query_emb, const std::vector<Tensor>& doc_embs) const;
};

class CrossEncoder {
public:
    Linear input_proj;
    Linear hidden_proj;
    Linear score_head;
    int64_t hidden_size;

    CrossEncoder(int64_t input_dim = 768, int64_t hidden_dim = 512);
    Tensor forward(const Tensor& query_emb, const Tensor& doc_emb, const Tensor& query_mask, const Tensor& doc_mask) const;
    float score_pair(const Tensor& query_emb, const Tensor& doc_emb) const;
    Tensor score_batch(const Tensor& query_emb, const std::vector<Tensor>& doc_embs, const Tensor& query_mask) const;
};

} // namespace multimodal
} // namespace quant
