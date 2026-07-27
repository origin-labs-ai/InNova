#pragma once
// ============================================================================
// PILLAR 5: MoMMoE Block — Mixture of Multimodal Experts (The Gemini Killer)
// ============================================================================
// WHY: Gemini-style multimodal models need tokens from different modalities
// (text, vision, audio) to attend to each other. MoMMoE routes each token
// to specialized experts based on its modality, while cross-modal attention
// allows information flow between modalities.
//
// ARCHITECTURE:
//   Input tokens → Modality Classifier → Top-2 Router → Expert FFNs
//   Cross-Modal Attention: Text attends to Vision, Vision attends to Text
//
// LOAD BALANCING: Auxiliary loss + z-loss prevent expert collapse.
// Without these, the router would always pick the same2 experts.
//
// TOKEN BUDGET (840B total for14B MoMMoE):
//   4 text experts:  ~105B tokens each (70/30 real/synthetic)
//   1 vision expert: ~105B tokens (80/20)
//   1 audio expert:  ~105B tokens (60/40)
//   1 image expert:  ~105B tokens (80/20)
//   1 video expert:  ~105B tokens (70/30)
// ============================================================================

#include "oil/tensor.h"
#include "oil/transformer.h"
#include "oil/moe_variants.h"
#include "oil/kv_cache.h"
#include "oil/math.h"
#include <vector>
#include <cstdint>
#include <string>

namespace oil {
namespace moe {

// ============================================================================
// Modality Types — What kind of data this token represents
// ============================================================================

enum class Modality : uint8_t {
    TEXT = 0,
    VISION,         // Image patches (ViT-style)
    AUDIO,          // Audio codec tokens (EnCodec/DAC)
    IMAGE_GEN,      // Latent diffusion tokens
    VIDEO_GEN,      // Video frame tokens
    OCR,            // Text-in-image tokens
    EMBEDDING,      // Contrastive embedding pairs
    CROSS_MODAL,    // Mixed-modality attention output
    META_COGNITION, // Self-referential reasoning tokens
    COUNT
};

inline const char* modality_name(Modality m) {
    static const char* names[] = {
        "TEXT", "VISION", "AUDIO", "IMAGE_GEN", "VIDEO_GEN",
        "OCR", "EMBEDDING", "CROSS_MODAL", "META_COGNITION"
    };
    return names[static_cast<int>(m)];
}

// ============================================================================
// TopKRouter — Modality-aware Top-2 expert routing with load balancing
// ============================================================================
// Routes each token to 2 experts out of 8.
// Load balance loss: encourages uniform expert utilization.
// z-loss: prevents router logits from growing too large.
// ============================================================================

struct RouterOutput {
    Tensor gates;           // {B*S, num_experts} — softmax probabilities
    Tensor indices;         // {B*S, top_k} — selected expert indices
    Tensor weights;         // {B*S, top_k} — routing weights (sum to 1)
    Tensor modality_probs;  // {B*S, num_modalities} — modality classification
    Modality dominant_modality;
    float load_balance_loss;
    float z_loss;
    int64_t num_activated_experts;
};

class TopKRouter {
public:
    TopKRouter(int64_t hidden_size, int64_t num_experts, int64_t top_k,
               int64_t num_modalities = static_cast<int64_t>(Modality::COUNT));

    RouterOutput forward(const Tensor& x, const Tensor* modality_hints = nullptr);

    // Load balance loss: f = N * sum(f_i * P_i)
    // f_i = fraction of tokens routed to expert i
    // P_i = average routing probability for expert i
    float load_balance_loss(const Tensor& gates, const Tensor& indices) const;

    // z-loss: regularizer to keep router logits small
    float z_loss(const Tensor& logits) const;

    // Modality classifier: predicts modality from hidden state
    Modality classify_modality(const Tensor& hidden) const;

    // Access weights
    Tensor& gate_weight() { return gate_weight_; }
    Tensor& modality_head() { return modality_head_; }
    const Tensor& gate_weight() const { return gate_weight_; }
    const Tensor& modality_head() const { return modality_head_; }

    int64_t num_experts() const { return num_experts_; }
    int64_t top_k() const { return top_k_; }

private:
    int64_t hidden_size_;
    int64_t num_experts_;
    int64_t top_k_;
    int64_t num_modalities_;

    Tensor gate_weight_;     // {hidden_size, num_experts}
    Tensor modality_head_;   // {hidden_size, num_modalities}

    // Running statistics for load balance monitoring
    std::vector<int64_t> expert_counts_;
    int64_t total_tokens_routed_ = 0;
};

// ============================================================================
// CrossModalAttention — Allows tokens from different modalities to attend
// ============================================================================
// Standard multi-head attention but with modality-aware masking.
// Text tokens can attend to vision tokens and vice versa.
// This is how the model learns cross-modal relationships.
// ============================================================================

class CrossModalAttention {
public:
    CrossModalAttention(int64_t hidden_size, int64_t num_heads,
                        int64_t head_dim = -1);

    // Forward: tokens from different modalities attend to each other
    Tensor forward(const Tensor& query_tokens, const Tensor& key_value_tokens,
                   const Tensor* attention_mask = nullptr);

    // Access projections
    Tensor& q_proj() { return q_proj_; }
    Tensor& k_proj() { return k_proj_; }
    Tensor& v_proj() { return v_proj_; }
    Tensor& o_proj() { return o_proj_; }
    const Tensor& q_proj() const { return q_proj_; }
    const Tensor& k_proj() const { return k_proj_; }
    const Tensor& v_proj() const { return v_proj_; }
    const Tensor& o_proj() const { return o_proj_; }

private:
    int64_t hidden_size_;
    int64_t num_heads_;
    int64_t head_dim_;

    Tensor q_proj_, k_proj_, v_proj_, o_proj_;
};

// ============================================================================
// ModalityExpertFFN — Single expert feed-forward network (Tensor-based)
// ============================================================================
// SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
// Each expert specializes in a specific modality/type of reasoning.
// ============================================================================

class ModalityExpertFFN {
public:
    ModalityExpertFFN();
    ModalityExpertFFN(int64_t hidden_size, int64_t ffn_hidden);
    ModalityExpertFFN(int64_t hidden_size, int64_t ffn_hidden, Modality specialization);

    Tensor forward(const Tensor& x) const;

    Tensor& gate_proj() { return gate_proj_; }
    Tensor& up_proj() { return up_proj_; }
    Tensor& down_proj() { return down_proj_; }
    const Tensor& gate_proj() const { return gate_proj_; }
    const Tensor& up_proj() const { return up_proj_; }
    const Tensor& down_proj() const { return down_proj_; }

    Modality specialization() const { return specialization_; }

private:
    Tensor gate_proj_;
    Tensor up_proj_;
    Tensor down_proj_;
    Modality specialization_;
};

// ============================================================================
// MoMBlock — Single Multimodal MoE Transformer Block
// ============================================================================
// Architecture:
//   1. Attention Norm → Multi-Head Self-Attention
//   2. Cross-Modal Attention (optional)
//   3. MoE Norm → TopK Router → Expert FFNs → Combine
//
// This is the fundamental building block of the MoMMoE.
// ============================================================================

class MoMBlock {
public:
    MoMBlock(int64_t hidden_size, int64_t num_heads, int64_t ffn_hidden,
             int64_t num_experts, int64_t top_k,
             int64_t num_modalities = static_cast<int64_t>(Modality::COUNT));

    // Forward pass through the MoM block
    struct MoMOutput {
        Tensor hidden;               // Output hidden states
        RouterOutput routing;        // Expert routing info
        Tensor cross_modal_output;   // Output of cross-modal attention
    };

    MoMOutput forward(const Tensor& x, const Tensor* modality_hints = nullptr,
                      KVCache* cache = nullptr, int64_t layer_idx = 0);

    // Access sub-modules
    RMSNorm& attention_norm() { return attention_norm_; }
    const RMSNorm& attention_norm() const { return attention_norm_; }
    Attention& self_attention() { return self_attention_; }
    const Attention& self_attention() const { return self_attention_; }
    CrossModalAttention& cross_modal_attn() { return cross_modal_attn_; }
    const CrossModalAttention& cross_modal_attn() const { return cross_modal_attn_; }
    RMSNorm& moe_norm() { return moe_norm_; }
    const RMSNorm& moe_norm() const { return moe_norm_; }
    TopKRouter& router() { return router_; }
    const TopKRouter& router() const { return router_; }
    std::vector<ModalityExpertFFN>& experts() { return experts_; }      
    const std::vector<ModalityExpertFFN>& experts() const { return experts_; }

private:
    int64_t hidden_size_;
    int64_t num_experts_;

    RMSNorm attention_norm_;
    Attention self_attention_;
    CrossModalAttention cross_modal_attn_;
    RMSNorm moe_norm_;
    TopKRouter router_;
    std::vector<ModalityExpertFFN> experts_;
};

// ============================================================================
// MoMMoE — Full Mixture of Multimodal Experts Model
// ============================================================================
// Stack of MoMBlocks with shared embeddings and output head.
// This is the complete14B parameter model.
//
// Expert allocation:
//   Text: 4 experts (70/30 real/synthetic)
//   Vision: 1 expert (80/20)
//   Audio: 1 expert (60/40)
//   Image Gen: 1 expert (80/20)
//   Video Gen: 1 expert (70/30)
//   Total: 8 experts, Top-2 routing → ~3.5B active per token
// ============================================================================

struct MoMMoEConfig {
    int64_t hidden_size = 2048;
    int64_t num_heads = 16;
    int64_t ffn_hidden = 8192;
    int64_t num_layers = 24;
    int64_t num_experts = 8;
    int64_t top_k = 2;
    int64_t vocab_size = 32000;
    int64_t max_seq_len = 1048576; // 1M tokens
    int64_t num_modalities = static_cast<int64_t>(Modality::COUNT);

    // Expert-modality mapping
    int64_t text_experts = 4;
    int64_t vision_experts = 1;
    int64_t audio_experts = 1;
    int64_t image_gen_experts = 1;
    int64_t video_gen_experts = 1;
};

class MoMMoE {
public:
    explicit MoMMoE(const MoMMoEConfig& cfg = MoMMoEConfig{});

    // Forward pass
    Tensor forward(const Tensor& input_ids, const Tensor* modality_hints = nullptr);

    // Generate text autoregressively
    Tensor generate(const Tensor& prompt_ids, int64_t max_new_tokens,
                    float temperature = 1.0f, int64_t top_k = 50);

    // Access layers
    std::vector<MoMBlock>& layers() { return layers_; }
    const MoMMoEConfig& config() const { return cfg_; }

    // Telemetry: total parameters
    int64_t total_parameters() const;
    int64_t active_parameters() const; // ~3.5B with Top-2

    // Expert utilization stats
    std::vector<float> expert_utilization() const;

private:
    MoMMoEConfig cfg_;
    Tensor embeddings_;     // {vocab_size, hidden_size}
    std::vector<MoMBlock> layers_;
    Tensor final_norm_;     // RMSNorm
    Tensor lm_head_;        // {hidden_size, vocab_size} (tied with embeddings)
};

} // namespace moe
} // namespace oil
