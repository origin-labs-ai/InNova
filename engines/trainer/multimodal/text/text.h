#pragma once
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/transformer.h"
#include "quant/tokenizer.h"
#include <vector>
#include <string>
#include <cmath>

namespace quant {
namespace multimodal {

struct TextConfig {
    int64_t vocab_size = 32000;
    int64_t hidden_size = 768;
    int64_t num_layers = 6;
    int64_t num_heads = 8;
    int64_t head_dim = 96;
    int64_t ffn_hidden = 3072;
    int64_t max_seq_len = 2048;
    int64_t num_classes = 0;
    float norm_eps = 1e-5f;
    float dropout = 0.1f;
};

class SinusoidalPositionalEncoding {
public:
    Tensor pe;
    int64_t max_len;
    int64_t dim;

    SinusoidalPositionalEncoding(int64_t max_len, int64_t dim);
    Tensor forward(int64_t seq_len) const;
};

class LearnablePositionalEncoding {
public:
    Tensor weight;
    int64_t max_len;
    int64_t dim;

    LearnablePositionalEncoding(int64_t max_len, int64_t dim);
    Tensor forward(int64_t seq_len) const;
};

enum class PoolingStrategy { CLS_TOKEN, MEAN, MAX };

class SequenceClassifier {
public:
    Linear dense;
    Linear classifier;
    int64_t hidden_size;
    int64_t num_classes;
    float dropout_ratio;

    SequenceClassifier(int64_t hidden_size, int64_t num_classes, float dropout = 0.1f);
    Tensor forward(const Tensor& pooled) const;
    Tensor predict_proba(const Tensor& pooled) const;
    int64_t predict_class(const Tensor& pooled) const;
    size_t param_count() const;
    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);
};

class BPEEncoder {
public:
    BPETokenizer tokenizer;
    int64_t max_seq_len;

    explicit BPEEncoder(int64_t max_seq_len = 2048);
    std::vector<int> encode_text(const std::string& text);
    Tensor tokenize_to_tensor(const std::string& text);
    Tensor compute_attention_mask(const Tensor& token_ids) const;
    std::string decode_tokens(const Tensor& token_ids);
};

Tensor apply_cls_pool(const Tensor& token_embs, int64_t cls_pos = 0);
Tensor apply_mean_pool(const Tensor& token_embs, const Tensor& mask);
Tensor apply_max_pool(const Tensor& token_embs, const Tensor& mask);

class TextEncoder {
public:
    quant::Embedding embedding;
    SinusoidalPositionalEncoding pos_enc;
    std::vector<quant::TransformerBlock> blocks;
    quant::RMSNorm final_norm;
    int64_t hidden_size;
    int64_t num_layers;
    int64_t num_heads;

    explicit TextEncoder(const TextConfig& cfg = TextConfig());
    Tensor encode(const Tensor& tokens) const;
    Tensor encode_with_pooling(const Tensor& tokens, const Tensor& mask, PoolingStrategy strategy) const;
    Tensor classify(const Tensor& tokens, const SequenceClassifier& classifier, const Tensor& mask, PoolingStrategy strategy) const;
    size_t param_count() const;
};

class TextDecoder {
public:
    quant::Embedding embedding;
    SinusoidalPositionalEncoding pos_enc;
    std::vector<quant::TransformerBlock> blocks;
    quant::Linear lm_head;
    quant::RMSNorm final_norm;
    int64_t hidden_size;
    int64_t vocab_size;
    int64_t num_layers;

    explicit TextDecoder(const TextConfig& cfg = TextConfig());
    Tensor decode(const Tensor& hidden_state, int64_t max_len) const;
    Tensor generate(const Tensor& tokens, int64_t max_new_tokens, const Tensor& mask) const;
    size_t param_count() const;
};

} // namespace multimodal
} // namespace quant
