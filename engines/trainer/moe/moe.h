#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "quant/tensor.h"
#include "quant/transformer.h"

namespace quant {
namespace moe {

enum class ModalityType {
    TEXT = 0,
    VISION,
    IMAGE_GEN,
    VIDEO_GEN,
    AUDIO,
    OCR,
    EMBEDDING,
    CROSS_MODAL,
    META_COGNITION,
    UNKNOWN
};

inline const char* modality_name(ModalityType m) {
    switch (m) {
        case ModalityType::TEXT: return "TEXT";
        case ModalityType::VISION: return "VISION";
        case ModalityType::IMAGE_GEN: return "IMAGE_GEN";
        case ModalityType::VIDEO_GEN: return "VIDEO_GEN";
        case ModalityType::AUDIO: return "AUDIO";
        case ModalityType::OCR: return "OCR";
        case ModalityType::EMBEDDING: return "EMBEDDING";
        case ModalityType::CROSS_MODAL: return "CROSS_MODAL";
        case ModalityType::META_COGNITION: return "META_COGNITION";
        default: return "UNKNOWN";
    }
}

struct MoEConfig {
    int64_t num_experts = 16;
    int64_t top_k = 3;
    int64_t expert_hidden_size = 2048;
    float load_balance_coef = 0.01f;
    float z_loss_coef = 0.001f;
    // Modality allocation: how many experts per modality
    int64_t text_experts = 4;
    int64_t vision_experts = 3;
    int64_t image_gen_experts = 2;
    int64_t video_gen_experts = 2;
    int64_t audio_experts = 2;
    int64_t ocr_experts = 1;
    int64_t cross_modal_experts = 2;
};

struct RouterOutput {
    Tensor gates;       // {B, S, num_experts}
    Tensor indices;     // {B, S, top_k}
    Tensor weights;     // {B, S, top_k}
    Tensor modality_probs; // {B, S, num_modalities}
    ModalityType dominant_modality;
    float load_balance_loss;
    float z_loss;
};

class ModalityClassifier {
public:
    ModalityClassifier(int64_t hidden_size, int64_t num_modalities);
    Tensor forward(const Tensor& x);
    ModalityType predict_modality(const Tensor& logits);
    Tensor modality_head;
    int64_t hidden_size;
    int64_t num_modalities;
};

class MoERouter {
public:
    explicit MoERouter(int64_t hidden_size, const MoEConfig& cfg);
    RouterOutput forward(const Tensor& x, const Tensor& modality_hints);
    Tensor gate_weight;
    ModalityClassifier modality_cls;
private:
    MoEConfig config_;
};

class MoEFFN {
public:
    explicit MoEFFN(const MoEConfig& cfg, const TransformerConfig& tcfg);
    Tensor forward(const Tensor& x, const RouterOutput& routing);
    std::vector<FFN> experts;
    // Expert-to-modality mapping
    std::vector<ModalityType> expert_modalities;
    float last_z_loss = 0.0f;
private:
    MoEConfig config_;
};

// Cross-modal attention: allows different modality tokens to attend each other
class CrossModalAttention {
public:
    CrossModalAttention(int64_t hidden_size, int64_t num_heads);
    Tensor forward(const Tensor& query_modality, const Tensor& key_value_modality);
    Tensor q_proj, k_proj, v_proj, o_proj;
    int64_t hidden_size, num_heads, head_dim;
};

// Mixture of Multimodal Experts — the core AGI routing block
class MoMBlock {
public:
    MoMBlock(const MoEConfig& moe_cfg, const TransformerConfig& tcfg);
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx);
    RMSNorm attention_norm;
    Attention attention;
    RMSNorm moe_norm;
    MoERouter router;
    MoEFFN moe_ffn;
    CrossModalAttention cross_attn;
};

} // namespace moe
} // namespace quant
