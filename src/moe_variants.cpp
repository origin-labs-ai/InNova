#include "oil/moe_variants.h"
#include "oil/random.h"
#include "oil/autograd_functions.h"
#include "oil/simd_math.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <memory>
#include <atomic>

namespace oil {
namespace moe {

// ========================================================================
// Hash utility
// ========================================================================

int64_t hash_token(int64_t token_id, int64_t range) {
    uint64_t h = (uint64_t)token_id * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 37;
    h *= 0xBF58476D1CE4E5B9ULL;
    return (int64_t)(h % (uint64_t)range);
}

// ========================================================================
// MoE variant name lookup
// ========================================================================

const char* moe_variant_name(MoEVariant v) {
    switch (v) {
        case MoEVariant::SPARSE_TOP1: return "SPARSE_TOP1";
        case MoEVariant::SPARSE_TOP2: return "SPARSE_TOP2";
        case MoEVariant::SPARSE_TOPK: return "SPARSE_TOPK";
        case MoEVariant::SOFT_MIXTURE: return "SOFT_MIXTURE";
        case MoEVariant::HIERARCHICAL: return "HIERARCHICAL";
        case MoEVariant::MOMOE: return "MOMOE";
        case MoEVariant::EXPERT_CHOICE: return "EXPERT_CHOICE";
        case MoEVariant::HASH_ROUTED: return "HASH_ROUTED";
        case MoEVariant::CROSS_LAYER: return "CROSS_LAYER";
        case MoEVariant::MULTIMODAL: return "MULTIMODAL";
        case MoEVariant::MMOE: return "MMOE";
        case MoEVariant::DEEPSEEK_MOE: return "DEEPSEEK_MOE";
        case MoEVariant::BASE_LAYER: return "BASE_LAYER";
        case MoEVariant::DENSE_MOE: return "DENSE_MOE";
        case MoEVariant::SHARED_EXPERT: return "SHARED_EXPERT";
        case MoEVariant::RESIDUAL_MOE: return "RESIDUAL_MOE";
        case MoEVariant::GATING_DROPOUT: return "GATING_DROPOUT";
        case MoEVariant::DOMAIN_MOE: return "DOMAIN_MOE";
        case MoEVariant::PRODUCT_KEY: return "PRODUCT_KEY";
        case MoEVariant::ATTENTION_MOE: return "ATTENTION_MOE";
        case MoEVariant::MLA_MOE: return "MLA_MOE";
        case MoEVariant::MAMBA_MOE: return "MAMBA_MOE";
        case MoEVariant::QUANTIZED_INT8_MOE: return "QUANTIZED_INT8_MOE";
        case MoEVariant::SPARK_MOE: return "SPARK_MOE";
        case MoEVariant::OIL1_MOE: return "OIL1_MOE";
        case MoEVariant::OIL8_MOE: return "OIL8_MOE";
        case MoEVariant::OIL4_MOE: return "OIL4_MOE";
        default: return "UNKNOWN";
    }
}

// ========================================================================
// ExpertFFN
// ========================================================================

ExpertFFN::ExpertFFN() : activation(Activation::SiLU) {}

ExpertFFN::ExpertFFN(int64_t hidden_size, int64_t ffn_hidden, Activation act)
    : gate_proj(hidden_size, ffn_hidden),
      up_proj(hidden_size, ffn_hidden),
      down_proj(ffn_hidden, hidden_size),
      activation(act) {}

Tensor ExpertFFN::forward(const Tensor& x) const {
    Tensor gate = gate_proj.forward(x);
    Tensor up = up_proj.forward(x);
    Tensor act_out({gate.shape()});
    if (activation == Activation::SiLU) {
        math::silu(gate, act_out);
    } else if (activation == Activation::GELU) {
        math::gelu(gate, act_out);
    } else {
        math::relu(gate, act_out);
    }
    Tensor gated({gate.shape()});
    math::mul(act_out, up, gated);
    return down_proj.forward(gated);
}

// ========================================================================
// Softmax with Top-K extraction
// ========================================================================

Tensor softmax_with_topk(const Tensor& logits, int64_t k, Tensor& indices_out, Tensor& weights_out) {
    int64_t T = logits.dim(0);
    int64_t E = logits.dim(1);
    Tensor probs({T, E});
    const float* l = logits.data<float>();
    float* p = probs.data<float>();
    indices_out = Tensor({T, k}, DType::I64);
    weights_out = Tensor({T, k});
    int64_t* idx = indices_out.data<int64_t>();
    float* w = weights_out.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        float maxv = l[t * E];
        for (int64_t e = 1; e < E; ++e)
            if (l[t * E + e] > maxv) maxv = l[t * E + e];
        float sum = 0.0f;
        for (int64_t e = 0; e < E; ++e) {
            float v = std::exp(l[t * E + e] - maxv);
            p[t * E + e] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (int64_t e = 0; e < E; ++e)
            p[t * E + e] *= inv;

        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e = 0; e < E; ++e)
            scored.push_back({p[t * E + e], e});
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t j = 0; j < k; ++j) {
            idx[t * k + j] = scored[j].second;
            w[t * k + j] = scored[j].first;
        }
    }
    return probs;
}

// ========================================================================
// Load balancing loss calculation
// ========================================================================

float compute_load_balance_loss(const Tensor& router_logits, const Tensor& expert_indices, int64_t num_experts) {
    int64_t T = router_logits.dim(0);
    int64_t K = expert_indices.dim(1);
    const int64_t* idx = expert_indices.data<int64_t>();
    const float* w = router_logits.data<float>();
    std::vector<double> f_i(num_experts, 0.0);
    std::vector<double> P_i(num_experts, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = (int64_t)idx[t * K + k];
            if (e >= 0 && e < num_experts) {
                f_i[e] += 1.0;
                P_i[e] += w[t * num_experts + e];
            }
        }
    }
    double loss = 0.0;
    for (int64_t e = 0; e < num_experts; ++e) {
        f_i[e] /= (double)T;
        P_i[e] /= (double)T;
        loss += f_i[e] * P_i[e];
    }
    return (float)(loss * (double)num_experts);
}

float compute_z_loss(const Tensor& expert_output_norms) {
    const float* d = expert_output_norms.data<float>();
    float sum = 0.0f;
    for (int64_t i = 0; i < expert_output_norms.numel(); ++i)
        sum += d[i] * d[i];
    return sum;
}

// ========================================================================
// ExpertFFN loader
// ========================================================================

std::vector<ExpertFFN> create_experts(int64_t count, int64_t hidden, int64_t ffn_hidden, Activation act) {
    std::vector<ExpertFFN> exps;
    exps.reserve(count);
    for (int64_t i = 0; i < count; ++i)
        exps.emplace_back(hidden, ffn_hidden, act);
    return exps;
}

// ========================================================================
// 1. SPARSE MoE
// ========================================================================

SparseMoE::SparseMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput SparseMoE::forward(const Tensor& x, bool training) {
    (void)training;
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;

    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router_weight.forward(x_flat);

    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, &dropped);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// ========================================================================
// 2. SOFT MoE (Dense Mixture)
// ========================================================================

SoftMoE::SoftMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden)
{
    int64_t E = cfg.num_experts;
    int64_t S = cfg.num_slots_per_expert;
    int64_t ffn_hidden = cfg.expert_hidden_size;
    int64_t total_slots = E * S;
    input_mixing = Linear(hidden, total_slots);
    output_mixing = Linear(hidden, hidden);
    experts = create_experts(E, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput SoftMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t slots = config.num_slots_per_expert;
    int64_t total_slots = E * slots;

    Tensor x_flat = x.reshape({T, D});

    // Learn input-dependent slot assignments
    Tensor slot_weights = input_mixing.forward(x_flat);
    Tensor slot_softmax({T, total_slots});
    math::softmax(slot_weights, slot_softmax, 1);

    // Each expert processes weighted combination of inputs
    Tensor expert_in({E, D});
    expert_in.zero_();
    float* ei = expert_in.data<float>();
    const float* xd = x_flat.data<float>();
    const float* sw = slot_softmax.data<float>();

    for (int64_t e = 0; e < E; ++e) {
        for (int64_t s = 0; s < slots; ++s) {
            int64_t slot_idx = e * slots + s;
            for (int64_t t = 0; t < T; ++t) {
                float w = sw[t * total_slots + slot_idx];
                for (int64_t d = 0; d < D; ++d)
                    ei[e * D + d] += w * xd[t * D + d];
            }
        }
    }

    // Expert forward
    Tensor expert_out({E, D});
    float* eo = expert_out.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        Tensor inp({1, D});
        std::memcpy(inp.data<float>(), ei + e * D, D * sizeof(float));
        Tensor out = experts[(size_t)e].forward(inp);
        std::memcpy(eo + e * D, out.data<float>(), D * sizeof(float));
    }

    // Combine: output[t] = Σ_e Σ_s slot_weight[t, e, s] × expert_out[e]
    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            for (int64_t s = 0; s < slots; ++s) {
                float w = sw[t * total_slots + e * slots + s];
                for (int64_t d = 0; d < D; ++d)
                    od[t * D + d] += w * eo[e * D + d];
            }
        }
    }

    // Output projection
    Tensor final = output_mixing.forward(output);

    MoEOutput out;
    out.output = final.reshape({B, S, D});
    out.router_logits = slot_weights;
    out.load_balance_loss = 0.0f;
    out.z_loss = config.z_loss_coef * compute_z_loss(expert_out) / (float)T;
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 3. HIERARCHICAL MoE
// ========================================================================

HierarchicalMoE::HierarchicalMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      group_router(hidden, cfg.num_groups)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    expert_groups.resize((size_t)cfg.num_groups);
    expert_routers.reserve(cfg.num_groups);
    for (int64_t g = 0; g < cfg.num_groups; ++g) {
        expert_groups[(size_t)g] = create_experts(cfg.experts_per_group, hidden, ffn_hidden, Activation::SiLU);
        expert_routers.emplace_back(hidden, cfg.experts_per_group);
    }
}

MoEOutput HierarchicalMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t G = config.num_groups;
    int64_t TG = config.top_groups;
    int64_t E = config.experts_per_group;
    int64_t K = config.top_experts_per_group;

    Tensor x_flat = x.reshape({T, D});

    // Level 1: group selection
    Tensor group_logits = group_router.forward(x_flat);
    Tensor g_indices, g_weights;
    Tensor g_probs = softmax_with_topk(group_logits, TG, g_indices, g_weights);

    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    const float* xd = x_flat.data<float>();
    int64_t* gi = g_indices.data<int64_t>();
    float* gw = g_weights.data<float>();

    float zl = 0.0f;

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t gk = 0; gk < TG; ++gk) {
            int64_t g = gi[t * TG + gk];
            float gw_val = gw[t * TG + gk];
            if (g < 0 || g >= G || gw_val <= 0.0f) continue;

            // Level 2: expert selection within group
            Tensor expert_logits = expert_routers[(size_t)g].forward(
                x_flat.slice(0, t, t + 1));
            Tensor e_indices, e_weights;
            Tensor e_probs = softmax_with_topk(expert_logits, K, e_indices, e_weights);

            int64_t* ei = e_indices.data<int64_t>();
            float* ew = e_weights.data<float>();

            for (int64_t ek = 0; ek < K; ++ek) {
                int64_t e = ei[ek];
                float ew_val = ew[ek];
                if (e < 0 || e >= E || ew_val <= 0.0f) continue;

                Tensor inp({1, D});
                std::memcpy(inp.data<float>(), xd + t * D, D * sizeof(float));
                Tensor eout = expert_groups[(size_t)g][(size_t)e].forward(inp);
                const float* ed = eout.data<float>();
                float mix = gw_val * ew_val;
                for (int64_t d = 0; d < D; ++d) {
                    float v = mix * ed[d];
                    od[t * D + d] += v;
                    zl += v * v;
                }
            }
        }
    }

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = group_logits;
    out.load_balance_loss = compute_load_balance_loss(group_logits, g_indices, G);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = TG * K;
    return out;
}

// ========================================================================
// 4. MoMoE — Mixture of Mixture of Experts
// ========================================================================

MoMoE::MoMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      primary_router(hidden, cfg.num_groups)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    groups.resize((size_t)cfg.num_groups);
    secondary_routers.reserve(cfg.num_groups);
    for (int64_t g = 0; g < cfg.num_groups; ++g) {
        groups[(size_t)g] = create_experts(cfg.experts_per_group, hidden, ffn_hidden, Activation::SiLU);
        secondary_routers.emplace_back(hidden, cfg.experts_per_group);
    }
}

MoEOutput MoMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t G = config.num_groups;
    int64_t TG = config.top_groups;
    int64_t E = config.experts_per_group;
    int64_t TK = config.top_experts_per_group;

    Tensor x_flat = x.reshape({T, D});

    // Primary routing: tokens → groups
    Tensor primary_logits = primary_router.forward(x_flat);
    Tensor g_indices, g_weights;
    Tensor g_probs = softmax_with_topk(primary_logits, TG, g_indices, g_weights);

    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    const float* xd = x_flat.data<float>();
    int64_t* gi = g_indices.data<int64_t>();
    float* gw = g_weights.data<float>();

    float zl = 0.0f;

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t gk = 0; gk < TG; ++gk) {
            int64_t g = gi[t * TG + gk];
            float gw_val = gw[t * TG + gk];
            if (g < 0 || g >= G || gw_val <= 0.0f) continue;

            // Secondary routing: within group MoE
            Tensor sec_logits = secondary_routers[(size_t)g].forward(
                x_flat.slice(0, t, t + 1));
            Tensor e_indices, e_weights;
            softmax_with_topk(sec_logits, TK, e_indices, e_weights);

            int64_t* ei = e_indices.data<int64_t>();
            float* ew = e_weights.data<float>();

            for (int64_t ek = 0; ek < TK; ++ek) {
                int64_t e = ei[ek];
                float ew_val = ew[ek];
                if (e < 0 || e >= E || ew_val <= 0.0f) continue;

                Tensor inp({1, D});
                std::memcpy(inp.data<float>(), xd + t * D, D * sizeof(float));
                Tensor eout = groups[(size_t)g][(size_t)e].forward(inp);
                const float* ed = eout.data<float>();
                float mix = gw_val * ew_val;
                for (int64_t d = 0; d < D; ++d) {
                    float v = mix * ed[d];
                    od[t * D + d] += v;
                    zl += v * v;
                }
            }
        }
    }

    // Load balance: both primary and secondary
    float lb_loss = compute_load_balance_loss(primary_logits, g_indices, G);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = primary_logits;
    out.load_balance_loss = lb_loss;
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = TG * TK;
    return out;
}

// ========================================================================
// 5. EXPERT CHOICE MoE
// ========================================================================

ExpertChoiceMoE::ExpertChoiceMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput ExpertChoiceMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t capacity = std::min(T, (int64_t)(config.capacity_factor * T / E));
    if (capacity < 1) capacity = 1;

    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router_weight.forward(x_flat);
    const float* l = logits.data<float>();

    // Each expert picks its top-K tokens
    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    const float* xd = x_flat.data<float>();

    float zl = 0.0f;
    int64_t total_assigned = 0;

    for (int64_t e = 0; e < E; ++e) {
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(T);
        for (int64_t t = 0; t < T; ++t)
            scored.push_back({l[t * E + e], t});
        int64_t actual_cap = std::min(capacity, T);
        std::partial_sort(scored.begin(), scored.begin() + actual_cap, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<int64_t> chosen;
        for (int64_t k = 0; k < actual_cap; ++k)
            if (scored[k].first > 0.0f) chosen.push_back(scored[k].second);
        int64_t nt = (int64_t)chosen.size();
        if (nt == 0) continue;

        Tensor batch_input({nt, D});
        float* bi = batch_input.data<float>();
        for (int64_t i = 0; i < nt; ++i)
            std::memcpy(bi + i * D, xd + chosen[(size_t)i] * D, (size_t)D * sizeof(float));

        Tensor batch_output = experts[(size_t)e].forward(batch_input);
        const float* bo = batch_output.data<float>();
        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = chosen[(size_t)i];
            for (int64_t d = 0; d < D; ++d) {
                od[t * D + d] += bo[i * D + d];
                zl += bo[i * D + d] * bo[i * D + d];
            }
            total_assigned++;
        }
    }

    // Normalize by number of experts that processed each token
    for (int64_t t = 0; t < T; ++t) {
        float* row = od + t * D;
        float count = 0.0f;
        for (int64_t e = 0; e < E; ++e)
            if (l[t * E + e] > 0.0f) count += 1.0f;
        if (count > 1.0f) {
            float inv = 1.0f / count;
            for (int64_t d = 0; d < D; ++d)
                row[d] *= inv;
        }
    }

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = 0.0f;
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 6. HASH MoE
// ========================================================================

HashMoE::HashMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      num_buckets(cfg.num_experts * cfg.hash_bucket_size)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

int64_t HashMoE::hash_to_expert(int64_t token_id, int64_t num_experts, int64_t bucket_size) {
    uint64_t h = (uint64_t)token_id * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 37;
    h *= 0xBF58476D1CE4E5B9ULL;
    uint64_t bucket = h % (uint64_t)(num_experts * bucket_size);
    return (int64_t)(bucket / (uint64_t)bucket_size);
}

MoEOutput HashMoE::forward(const Tensor& x, const Tensor& token_ids) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;

    Tensor x_flat = x.reshape({T, D});
    const int64_t* ids = token_ids.data<int64_t>();
    const float* xd = x_flat.data<float>();

    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();

    // Group tokens by their assigned expert
    std::vector<std::vector<int64_t>> expert_tokens((size_t)E);
    for (int64_t t = 0; t < T; ++t) {
        int64_t e = hash_to_expert(ids[t], E, config.hash_bucket_size);
        if (e >= 0 && e < E)
            expert_tokens[(size_t)e].push_back(t);
    }

    float zl = 0.0f;
    for (int64_t e = 0; e < E; ++e) {
        auto& tokens = expert_tokens[(size_t)e];
        int64_t nt = (int64_t)tokens.size();
        if (nt == 0) continue;
        Tensor batch_input({nt, D});
        float* bi = batch_input.data<float>();
        for (int64_t i = 0; i < nt; ++i)
            std::memcpy(bi + i * D, xd + tokens[(size_t)i] * D, (size_t)D * sizeof(float));
        Tensor batch_output = experts[(size_t)e].forward(batch_input);
        const float* bo = batch_output.data<float>();
        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = tokens[(size_t)i];
            for (int64_t d = 0; d < D; ++d) {
                od[t * D + d] += bo[i * D + d];
                zl += bo[i * D + d] * bo[i * D + d];
            }
        }
    }

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.load_balance_loss = 0.0f;
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = (int64_t)expert_tokens.size();
    return out;
}

// ========================================================================
// 7. CROSS-LAYER MoE
// ========================================================================

CrossLayerMoE::CrossLayerMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    shared_experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
    layer_routers.reserve(cfg.num_shared_layers);
    for (int64_t i = 0; i < cfg.num_shared_layers; ++i)
        layer_routers.emplace_back(hidden, cfg.num_experts);
}

MoEOutput CrossLayerMoE::forward(const Tensor& x, int64_t layer_idx) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t K = config.top_k;

    size_t ridx = (size_t)(layer_idx % config.num_shared_layers);
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = layer_routers[ridx].forward(x_flat);

    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, shared_experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, nullptr);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// ========================================================================
// 8. MULTIMODAL MoE (MoMMoE)
// ========================================================================

MultiModalMoE::MultiModalMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts),
      modality_classifier(hidden, 9)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);

    // Build expert-to-modality map
    expert_modality_map.resize((size_t)cfg.num_experts);
    int64_t offset = 0;
    struct { int64_t count; int64_t mod; } mods[] = {
        {cfg.text_experts, 0}, {cfg.vision_experts, 1}, {cfg.image_gen_experts, 2},
        {cfg.video_gen_experts, 3}, {cfg.audio_experts, 4}, {cfg.ocr_experts, 5},
        {cfg.cross_modal_experts, 6}
    };
    for (auto& m : mods) {
        for (int64_t i = 0; i < m.count && offset < cfg.num_experts; ++i)
            expert_modality_map[(size_t)offset++] = m.mod;
    }
    // Fill remaining with text
    while (offset < cfg.num_experts)
        expert_modality_map[(size_t)offset++] = 0;
}

MoEOutput MultiModalMoE::forward(const Tensor& x, const Tensor& modality_hints) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t K = config.top_k;
    int64_t NM = num_modalities;

    Tensor x_flat = x.reshape({T, D});

    // Classify modality per token
    Tensor mod_logits = modality_classifier.forward(x_flat);
    Tensor mod_probs({T, NM});
    math::softmax(mod_logits, mod_probs, 1);

    // Compute routing logits with modality bias
    Tensor logits = router_weight.forward(x_flat);
    float* l = logits.data<float>();
    const float* mp = mod_probs.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            int64_t exp_mod = expert_modality_map[(size_t)e];
            l[t * E + e] += 10.0f * mp[t * NM + exp_mod];
        }
    }

    // Add external modality hints if provided
    if (modality_hints.numel() > 0) {
        const float* mh = modality_hints.data<float>();
        int64_t mh_T = modality_hints.numel();
        for (int64_t t = 0; t < T && t < mh_T; ++t) {
            int64_t hint_mod = (int64_t)mh[t];
            if (hint_mod >= 0 && hint_mod < NM) {
                for (int64_t e = 0; e < E; ++e) {
                    if (expert_modality_map[(size_t)e] == hint_mod)
                        l[t * E + e] += 5.0f;
                }
            }
        }
    }

    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, nullptr);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// ========================================================================
// Batched expert dispatch
// ========================================================================

Tensor moe_dispatch_batched(const Tensor& x_flat,
                            const std::vector<ExpertFFN>& experts,
                            const int64_t* indices, const float* weights,
                            int64_t T, int64_t K, int64_t E, int64_t D,
                            float* z_loss_out, int64_t* dropped_out) {
    Tensor output({T, D});
    output.zero_();

    std::vector<int> counts((size_t)E, 0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) counts[(size_t)e]++;
        }

    std::vector<std::vector<int64_t>> expert_tokens((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        expert_tokens[(size_t)e].reserve((size_t)counts[(size_t)e]);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) expert_tokens[(size_t)e].push_back(t);
        }

    const float* xd = x_flat.data<float>();
    float* od = output.data<float>();
    float zl = 0.0f;
    int64_t dropped = 0;

    for (int64_t e = 0; e < E; ++e) {
        int64_t nt = (int64_t)expert_tokens[(size_t)e].size();
        if (nt == 0) continue;

        Tensor batch_input({nt, D});
        float* bi = batch_input.data<float>();
        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = expert_tokens[(size_t)e][(size_t)i];
            std::memcpy(bi + i * D, xd + t * D, (size_t)D * sizeof(float));
        }

        Tensor batch_output = experts[(size_t)e].forward(batch_input);
        const float* bo = batch_output.data<float>();

        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = expert_tokens[(size_t)e][(size_t)i];
            float wgt = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                if (indices[t * K + k] == e) { wgt = weights[t * K + k]; break; }
            if (wgt <= 0.0f) { dropped++; continue; }
            for (int64_t d = 0; d < D; ++d) {
                float v = wgt * bo[i * D + d];
                od[t * D + d] += v;
                zl += v * v;
            }
        }
    }

    if (z_loss_out) *z_loss_out = zl;
    if (dropped_out) *dropped_out = dropped;
    return output;
}

// ========================================================================
// AVX2-optimized MoE kernels
// ========================================================================

namespace avx2 {

void moe_combine(float* output, const float* expert_outputs,
                 const float* weights, const int64_t* indices,
                 int64_t T, int64_t K, int64_t D) {
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            float w = weights[t * K + k];
            if (e < 0 || w <= 0.0f) continue;
            const float* src = expert_outputs + (e * T + t) * D;
            float* dst = output + t * D;
            for (int64_t d = 0; d < D; ++d)
                dst[d] += w * src[d];
        }
}

void moe_softmax_topk(float* probs, int64_t* indices, float* weights,
                      const float* logits, int64_t T, int64_t E, int64_t K) {
    for (int64_t t = 0; t < T; ++t) {
        const float* row = logits + t * E;
        float* p = probs + t * E;
        float maxv = row[0];
        for (int64_t e = 1; e < E; ++e)
            if (row[e] > maxv) maxv = row[e];

        float sum = 0.0f;
        int64_t e = 0;
#if defined(OIL_AVX2) || defined(__AVX2__)
        {
            __m256 maxv8 = _mm256_set1_ps(maxv);
            __m256 sumv = _mm256_setzero_ps();
            for (; e + 8 <= E; e += 8) {
                __m256 rv = _mm256_loadu_ps(row + e);
                __m256 ev = oil::simd::oil_exp_ps(_mm256_sub_ps(rv, maxv8));
                _mm256_storeu_ps(p + e, ev);
                sumv = _mm256_add_ps(sumv, ev);
            }
            float hsum[8];
            _mm256_storeu_ps(hsum, sumv);
            sum = hsum[0]+hsum[1]+hsum[2]+hsum[3]+hsum[4]+hsum[5]+hsum[6]+hsum[7];
        }
#endif
        for (; e < E; ++e) {
            p[e] = std::exp(row[e] - maxv);
            sum += p[e];
        }

        float inv = 1.0f / sum;
        for (int64_t e2 = 0; e2 < E; ++e2)
            p[e2] *= inv;

        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e2 = 0; e2 < E; ++e2)
            scored.push_back({p[e2], e2});
        std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t k = 0; k < K; ++k) {
            indices[t * K + k] = scored[k].second;
            weights[t * K + k] = scored[k].first;
        }
    }
}

void moe_load_balance(float* f_i, float* P_i,
                      const int64_t* indices, const float* weights,
                      int64_t T, int64_t K, int64_t E) {
    std::memset(f_i, 0, (size_t)E * sizeof(float));
    std::memset(P_i, 0, (size_t)E * sizeof(float));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) {
                f_i[e] += 1.0f;
                P_i[e] += weights[t * K + k];
            }
        }
    float inv_T = 1.0f / (float)T;
    for (int64_t e = 0; e < E; ++e) {
        f_i[e] *= inv_T;
        P_i[e] *= inv_T;
    }
}

} // namespace avx2

// ========================================================================
// 9. MMoE — Multi-gate Mixture of Experts
// ========================================================================

MMoE::MMoE(int64_t hidden_size, const MoEAllConfig& cfg)
    : hidden_size(hidden_size), config(cfg) {
    int64_t num_experts = cfg.num_experts;
    int64_t ffn_hidden = cfg.expert_hidden_size;
    int64_t num_tasks = cfg.num_tasks > 0 ? cfg.num_tasks : 1;

    experts.reserve(num_experts);
    for (int64_t i = 0; i < num_experts; i++)
        experts.emplace_back(hidden_size, ffn_hidden);

    task_gates.reserve(num_tasks);
    for (int64_t i = 0; i < num_tasks; i++)
        task_gates.emplace_back(hidden_size, num_experts);
}

MoEOutput MMoE::forward(const Tensor& x, int64_t task_id) {
    MoEOutput out;
    int64_t T = x.numel() / hidden_size;
    Tensor x_flat = x.reshape({T, hidden_size});
    int64_t E = config.num_experts;
    int64_t K = config.top_k > 0 ? config.top_k : 2;

    int64_t tid = (task_id >= 0 && task_id < (int64_t)task_gates.size()) ? task_id : 0;
    Tensor gate_logits = task_gates[tid].forward(x_flat);

    Tensor indices({T, K}, DType::I64);
    Tensor weights({T, K});
    Tensor probs = softmax_with_topk(gate_logits, K, indices, weights);

    out.router_logits = gate_logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = compute_load_balance_loss(gate_logits, indices, E);
    out.z_loss = compute_z_loss(gate_logits);

    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, hidden_size);

    out.output = output.reshape(x.shape());
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 10. DeepSeek-MoE (shared + routed experts)
// ========================================================================

DeepSeekMoE::DeepSeekMoE(int64_t hidden_size, const MoEAllConfig& cfg)
    : hidden_size(hidden_size), config(cfg),
      shared_expert(hidden_size, cfg.expert_hidden_size) {
    int64_t num_routed = cfg.num_routed_experts > 0 ? cfg.num_routed_experts : 8;
    int64_t ffn_hidden = cfg.expert_hidden_size;

    routed_experts.reserve(num_routed);
    for (int64_t i = 0; i < num_routed; i++)
        routed_experts.emplace_back(hidden_size, ffn_hidden);

    router_weight = Linear(hidden_size, num_routed);
    expert_biases.assign(num_routed, 0.0f);
}

MoEOutput DeepSeekMoE::forward(const Tensor& x, bool training) {
    MoEOutput out;
    int64_t T = x.numel() / hidden_size;
    Tensor x_flat = x.reshape({T, hidden_size});
    int64_t E = (int64_t)routed_experts.size();
    int64_t K = config.top_k > 0 ? config.top_k : 2;

    // Shared expert is always active
    Tensor shared_out = shared_expert.forward(x_flat);

    // Route to K experts
    Tensor gate_logits = router_weight.forward(x_flat);
    float* gl = gate_logits.data<float>();
    for (int64_t t = 0; t < T * E; t++)
        gl[t] += expert_biases[t % E];

    Tensor indices({T, K}, DType::I64);
    Tensor weights({T, K});
    Tensor probs = softmax_with_topk(gate_logits, K, indices, weights);

    // Bias update for load balancing (no auxiliary loss)
    if (training) {
        const int64_t* idx = indices.data<int64_t>();
        for (int64_t t = 0; t < T; t++)
            for (int64_t k = 0; k < K; k++)
                expert_biases[idx[t * K + k]] -= 0.001f;
        float mean_bias = 0;
        for (int64_t e = 0; e < E; e++) mean_bias += expert_biases[e];
        mean_bias /= E;
        for (int64_t e = 0; e < E; e++) expert_biases[e] -= mean_bias;
    }

    out.router_logits = gate_logits;
    out.expert_indices = indices;
    out.expert_weights = weights;

    Tensor routed_out = moe_dispatch_batched(x_flat, routed_experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, hidden_size);

    // Combine shared + routed
    Tensor output({T, hidden_size});
    float* od = output.data<float>();
    const float* sd = shared_out.data<float>();
    const float* rd = routed_out.data<float>();
    for (int64_t i = 0; i < T * hidden_size; ++i)
        od[i] = sd[i] + rd[i];

    out.output = output.reshape(x.shape());
    out.num_activated_experts = E + 1;
    return out;
}

} // namespace moe
} // namespace oil

// ========================================================================
// 11-24: Additional 14 MoE variant implementations
// ========================================================================

namespace oil {
namespace moe {

// 11. BASE Layer MoE
BaseLayerMoE::BaseLayerMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput BaseLayerMoE::forward(const Tensor& x, bool training) {
    (void)training;
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl, &dropped);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// 12. Dense MoE — all experts active
DenseMoE::DenseMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), gate(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput DenseMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = gate.forward(x_flat);
    Tensor probs({T, E});
    math::softmax(logits, probs, 1);
    const float* pd = probs.data<float>();
    const float* xd = x_flat.data<float>();
    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    float zl = 0.0f;
    for (int64_t e = 0; e < E; ++e) {
        Tensor batch_in({T, D});
        float* bi = batch_in.data<float>();
        for (int64_t t = 0; t < T; ++t)
            std::memcpy(bi + t * D, xd + t * D, (size_t)D * sizeof(float));
        Tensor expert_out = experts[(size_t)e].forward(batch_in);
        const float* eo = expert_out.data<float>();
        for (int64_t t = 0; t < T; ++t) {
            float w = pd[t * E + e];
            for (int64_t d = 0; d < D; ++d) {
                float v = w * eo[t * D + d];
                od[t * D + d] += v;
                zl += v * v;
            }
        }
    }
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = 0.0f;
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = E;
    return out;
}

// 13. Shared Expert MoE (standalone)
SharedExpertMoE::SharedExpertMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      shared_expert(hidden, cfg.expert_hidden_size),
      router(hidden, cfg.num_experts) {
    routed_experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput SharedExpertMoE::forward(const Tensor& x, bool training) {
    (void)training;
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor shared_out = shared_expert.forward(x_flat);
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    Tensor routed_out = moe_dispatch_batched(x_flat, routed_experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D);
    Tensor output({T, D});
    float* od = output.data<float>();
    const float* sd = shared_out.data<float>();
    const float* rd = routed_out.data<float>();
    for (int64_t i = 0; i < T * D; ++i)
        od[i] = sd[i] + rd[i];
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.num_activated_experts = E + 1;
    return out;
}

// 14. Residual MoE — overflow via residual
ResidualMoE::ResidualMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput ResidualMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor moe_out = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl, &dropped);
    Tensor output({T, D});
    float* od = output.data<float>();
    const float* xd = x_flat.data<float>();
    const float* md = moe_out.data<float>();
    for (int64_t i = 0; i < T * D; ++i)
        od[i] = xd[i] + md[i];
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// 15. Gating Dropout MoE
GatingDropoutMoE::GatingDropoutMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput GatingDropoutMoE::forward(const Tensor& x, bool training) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    if (training && dropout_rate > 0.0f) {
        float* ld = logits.data<float>();
        RNG rng(42);
        for (int64_t i = 0; i < T * E; ++i) {
            if (rng.uniform() < dropout_rate)
                ld[i] = 0.0f;
        }
    }
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl, &dropped);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// 16. Domain MoE
DomainMoE::DomainMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), domain_classifier(hidden, num_domains) {
    domain_experts.resize(num_domains);
    domain_routers.reserve(num_domains);
    int64_t per_domain = cfg.num_experts / num_domains;
    if (per_domain < 1) per_domain = 1;
    for (int64_t d = 0; d < num_domains; ++d) {
        domain_experts[(size_t)d] = create_experts(per_domain, hidden, cfg.expert_hidden_size, Activation::SiLU);
        domain_routers.emplace_back(hidden, per_domain);
    }
}

MoEOutput DomainMoE::forward(const Tensor& x, int64_t domain_id) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    Tensor x_flat = x.reshape({T, D});
    Tensor dom_logits = domain_classifier.forward(x_flat);
    Tensor dom_probs({T, num_domains});
    math::softmax(dom_logits, dom_probs, 1);
    int64_t K = config.top_k;
    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    const float* xd = x_flat.data<float>();
    float total_lb = 0.0f;
    for (int64_t d = 0; d < num_domains; ++d) {
        int64_t E = (int64_t)domain_experts[(size_t)d].size();
        Tensor r_logits = domain_routers[(size_t)d].forward(x_flat);
        Tensor indices, weights;
        softmax_with_topk(r_logits, K, indices, weights);
        float zl = 0.0f;
        Tensor d_out = moe_dispatch_batched(x_flat, domain_experts[(size_t)d],
            indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
        const float* dd = d_out.data<float>();
        const float* dp = dom_probs.data<float>();
        for (int64_t t = 0; t < T; ++t) {
            float w = dp[t * num_domains + d];
            for (int64_t i = 0; i < D; ++i)
                od[t * D + i] += w * dd[t * D + i];
        }
        total_lb += compute_load_balance_loss(r_logits, indices, E);
    }
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = dom_logits;
    out.load_balance_loss = total_lb / (float)num_domains;
    out.num_activated_experts = K * num_domains;
    return out;
}

// 17. Product Key MoE
ProductKeyMoE::ProductKeyMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      key_router_a(hidden, cfg.num_experts),
      key_router_b(hidden, cfg.num_experts) {
    int64_t half = cfg.num_experts / 2;
    if (half < 1) half = 1;
    experts_a = create_experts(half, hidden, cfg.expert_hidden_size, Activation::SiLU);
    experts_b = create_experts(cfg.num_experts - half, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput ProductKeyMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits_a = key_router_a.forward(x_flat);
    Tensor logits_b = key_router_b.forward(x_flat);
    Tensor indices_a, weights_a, indices_b, weights_b;
    softmax_with_topk(logits_a, K, indices_a, weights_a);
    softmax_with_topk(logits_b, K, indices_b, weights_b);
    int64_t Ea = (int64_t)experts_a.size();
    int64_t Eb = (int64_t)experts_b.size();
    float zl_a = 0, zl_b = 0;
    Tensor out_a = moe_dispatch_batched(x_flat, experts_a,
        indices_a.data<int64_t>(), weights_a.data<float>(), T, K, Ea, D, &zl_a);
    Tensor out_b = moe_dispatch_batched(x_flat, experts_b,
        indices_b.data<int64_t>(), weights_b.data<float>(), T, K, Eb, D, &zl_b);
    Tensor output({T, D});
    float* od = output.data<float>();
    const float* ad = out_a.data<float>();
    const float* bd = out_b.data<float>();
    for (int64_t i = 0; i < T * D; ++i)
        od[i] = ad[i] + bd[i];
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits_a;
    out.load_balance_loss = compute_load_balance_loss(logits_a, indices_a, Ea);
    out.z_loss = config.z_loss_coef * (zl_a + zl_b) / (float)T;
    out.num_activated_experts = 2 * K;
    return out;
}

// 18. Attention MoE — attention-based routing
AttentionMoE::AttentionMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      q_proj(hidden, cfg.num_experts),
      k_proj(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput AttentionMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor Q = q_proj.forward(x_flat);
    Tensor K_keys = k_proj.forward(x_flat);
    float scale = 1.0f / std::sqrt((float)D);
    const float* qd = Q.data<float>();
    const float* kd = K_keys.data<float>();
    Tensor logits({T, E});
    float* ld = logits.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            float dot = 0;
            for (int64_t d = 0; d < D; ++d)
                dot += qd[t * E + e % E] * kd[t * E + e];
            ld[t * E + e] = dot * scale;
        }
    }
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// 19. MLA MoE — Multi-Latent Attention MoE
MLAMoE::MLAMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      down_proj(hidden, latent_dim),
      up_proj(latent_dim, hidden),
      router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput MLAMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor latent = down_proj.forward(x_flat);
    Tensor recovered = up_proj.forward(latent);
    Tensor logits = router.forward(recovered);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(recovered, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// 20. Mamba MoE — SSM + MoE hybrid
MambaMoE::MambaMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      ssm_proj(hidden, state_dim),
      router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput MambaMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor ssm_out = ssm_proj.forward(x_flat);
    Tensor combined({T, D});
    float* cd = combined.data<float>();
    const float* xd = x_flat.data<float>();
    const float* sd = ssm_out.data<float>();
    for (int64_t t = 0; t < T; ++t)
        for (int64_t d = 0; d < D; ++d)
            cd[t * D + d] = xd[t * D + d] + sd[t * state_dim + d % state_dim];
    Tensor logits = router.forward(combined);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(combined, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// 21. Quantized INT8 MoE
QuantizedINT8MoE::QuantizedINT8MoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    expert_scales.resize(cfg.num_experts, 1.0f);
}

MoEOutput QuantizedINT8MoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    const float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        float s = expert_scales[(size_t)e];
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < D; ++d)
                const_cast<float*>(od)[t * D + d] *= s;
    }
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// 22. Spark MoE
SparkMoE::SparkMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    spark_scales.resize(cfg.num_experts, 1.0f);
}

MoEOutput SparkMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        float scale = spark_scales[(size_t)e];
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                float v = od[t * D + d] * scale;
                od[t * D + d] = (v > 0.1f) ? scale : ((v < -0.1f) ? -scale : 0.0f);
            }
        }
    }
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// 23. OIL1 MoE
Oil1MoE::Oil1MoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    oil1_scales.resize(cfg.num_experts, 1.0f);
}

MoEOutput Oil1MoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        float scale = oil1_scales[(size_t)e];
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                float v = od[t * D + d] * scale;
                od[t * D + d] = (v >= 0.0f) ? scale : -scale;
            }
        }
    }
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// 24. OIL8 MoE — codebook quantized experts
OIL8MoE::OIL8MoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    codebooks.resize(cfg.num_experts);
    for (int64_t e = 0; e < cfg.num_experts; ++e) {
        codebooks[(size_t)e].resize(256, 0.0f);
        for (int i = 0; i < 256; ++i)
            codebooks[(size_t)e][(size_t)i] = (float)(i - 128) * 0.01f;
    }
}

MoEOutput OIL8MoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        auto& cb = codebooks[(size_t)e];
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                float v = od[t * D + d];
                int idx = (int)std::clamp((int)(v * 100.0f + 128), 0, 255);
                od[t * D + d] = cb[(size_t)idx];
            }
        }
    }
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// 25. OIL4 MoE — 4-bit codebook quantized experts
OIL4MoE::OIL4MoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    codebooks.resize(cfg.num_experts);
    for (int64_t e = 0; e < cfg.num_experts; ++e) {
        codebooks[(size_t)e].resize(16, 0.0f);
        for (int i = 0; i < 16; ++i)
            codebooks[(size_t)e][(size_t)i] = (float)(i - 8) * 0.1f;
    }
}

MoEOutput OIL4MoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        auto& cb = codebooks[(size_t)e];
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                float v = od[t * D + d];
                int idx = (int)std::clamp((int)(v * 10.0f + 8), 0, 15);
                od[t * D + d] = cb[(size_t)idx];
            }
        }
    }
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// ========================================================================
// MoE Factory — maps variant enum to variant name/count
// ========================================================================

int64_t moe_variant_count() {
    return 26;
}

const char* moe_variant_name_by_index(int64_t index) {
    if (index < 0 || index >= moe_variant_count()) return "UNKNOWN";
    return moe_variant_name((MoEVariant)index);
}

std::unique_ptr<void, void(*)(void*)> create_moe_variant(MoEVariant variant, int64_t hidden, const MoEAllConfig& cfg) {
    switch (variant) {
        case MoEVariant::SPARSE_TOP1: case MoEVariant::SPARSE_TOP2: case MoEVariant::SPARSE_TOPK: {
            auto* p = new SparseMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<SparseMoE*>(v); }};
        }
        case MoEVariant::SOFT_MIXTURE: {
            auto* p = new SoftMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<SoftMoE*>(v); }};
        }
        case MoEVariant::HIERARCHICAL: {
            auto* p = new HierarchicalMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<HierarchicalMoE*>(v); }};
        }
        case MoEVariant::MOMOE: {
            auto* p = new MoMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MoMoE*>(v); }};
        }
        case MoEVariant::EXPERT_CHOICE: {
            auto* p = new ExpertChoiceMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<ExpertChoiceMoE*>(v); }};
        }
        case MoEVariant::HASH_ROUTED: {
            auto* p = new HashMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<HashMoE*>(v); }};
        }
        case MoEVariant::CROSS_LAYER: {
            auto* p = new CrossLayerMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<CrossLayerMoE*>(v); }};
        }
        case MoEVariant::MULTIMODAL: {
            auto* p = new MultiModalMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MultiModalMoE*>(v); }};
        }
        case MoEVariant::MMOE: {
            auto* p = new MMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MMoE*>(v); }};
        }
        case MoEVariant::DEEPSEEK_MOE: {
            auto* p = new DeepSeekMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<DeepSeekMoE*>(v); }};
        }
        case MoEVariant::BASE_LAYER: {
            auto* p = new BaseLayerMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<BaseLayerMoE*>(v); }};
        }
        case MoEVariant::DENSE_MOE: {
            auto* p = new DenseMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<DenseMoE*>(v); }};
        }
        case MoEVariant::SHARED_EXPERT: {
            auto* p = new SharedExpertMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<SharedExpertMoE*>(v); }};
        }
        case MoEVariant::RESIDUAL_MOE: {
            auto* p = new ResidualMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<ResidualMoE*>(v); }};
        }
        case MoEVariant::GATING_DROPOUT: {
            auto* p = new GatingDropoutMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<GatingDropoutMoE*>(v); }};
        }
        case MoEVariant::DOMAIN_MOE: {
            auto* p = new DomainMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<DomainMoE*>(v); }};
        }
        case MoEVariant::PRODUCT_KEY: {
            auto* p = new ProductKeyMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<ProductKeyMoE*>(v); }};
        }
        case MoEVariant::ATTENTION_MOE: {
            auto* p = new AttentionMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<AttentionMoE*>(v); }};
        }
        case MoEVariant::MLA_MOE: {
            auto* p = new MLAMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MLAMoE*>(v); }};
        }
        case MoEVariant::MAMBA_MOE: {
            auto* p = new MambaMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MambaMoE*>(v); }};
        }
        case MoEVariant::QUANTIZED_INT8_MOE: {
            auto* p = new QuantizedINT8MoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<QuantizedINT8MoE*>(v); }};
        }
        case MoEVariant::SPARK_MOE: {
            auto* p = new SparkMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<SparkMoE*>(v); }};
        }
        case MoEVariant::OIL1_MOE: {
            auto* p = new Oil1MoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<Oil1MoE*>(v); }};
        }
        case MoEVariant::OIL8_MOE: {
            auto* p = new OIL8MoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<OIL8MoE*>(v); }};
        }
        case MoEVariant::OIL4_MOE: {
            auto* p = new OIL4MoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<OIL4MoE*>(v); }};
        }
        default:
            return std::unique_ptr<void, void(*)(void*)>(nullptr, [](void*){});
    }
}

} // namespace moe
} // namespace oil
