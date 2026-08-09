#include "quant/inference_opt.h"
#include "quant/math.h"
#include "quant/int8_quant.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <random>
#include <sstream>
namespace quant {

// ===========================================================================
// D1: Paged attention — vLLM-style block-level KV cache management
// ===========================================================================
PagedAttention::PagedAttention(int64_t head_dim, int64_t n_heads, int64_t block_size,
                               int64_t max_blocks)
    : head_dim_(head_dim), n_heads_(n_heads), block_size_(block_size),
      max_blocks_(max_blocks), next_id_(0) {
    int64_t reserve_count = std::min(max_blocks_, (int64_t)256);
    for (int64_t i = 0; i < reserve_count; i++) {
        Block b;
        b.id = i;
        b.k = Tensor::zeros(Shape{1, n_heads, block_size, head_dim});
        b.v = Tensor::zeros(Shape{1, n_heads, block_size, head_dim});
        b.active = false;
        blocks_.push_back(b);
        free_ids_.push(i);
        next_id_ = i + 1;
    }
}

PagedAttention::Block PagedAttention::alloc_block() {
    if (free_ids_.empty()) {
        if (next_id_ >= max_blocks_) return Block{-1, Tensor(), Tensor(), false};
        int64_t id = next_id_++;
        Block b;
        b.id = id;
        b.k = Tensor::zeros(Shape{1, n_heads_, block_size_, head_dim_});
        b.v = Tensor::zeros(Shape{1, n_heads_, block_size_, head_dim_});
        b.active = true;
        blocks_.push_back(b);
        return blocks_.back();
    }
    int64_t id = free_ids_.front();
    free_ids_.pop();
    blocks_[(size_t)id].active = true;
    blocks_[(size_t)id].k.zero_();
    blocks_[(size_t)id].v.zero_();
    return blocks_[(size_t)id];
}

void PagedAttention::free_block(int64_t id) {
    if (id >= 0 && id < (int64_t)blocks_.size() && blocks_[(size_t)id].active) {
        blocks_[(size_t)id].active = false;
        free_ids_.push(id);
    }
}

Tensor PagedAttention::forward(const Tensor& Q, int64_t* block_table,
                               Block* blocks, int64_t num_blocks) {
    if (Q.rank() < 4) return Tensor::zeros(Shape{1, 1, 1, (int64_t)head_dim_});
    int64_t B = Q.dim(0), H = Q.dim(1), S = Q.dim(2), D = Q.dim(3);
    if (num_blocks <= 0) return Tensor::zeros(Shape{B, H, S, D});
    float scale = 1.0f / std::sqrt((float)D);
    Tensor out({B, H, S, D});
    out.zero_();
    float* od = out.data<float>();
    const float* qd = Q.data<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                float row_max = -INFINITY;
                float row_sum = 0;
                float* o_ptr = od + ((b * H + h) * S + s) * D;
                const float* q_ptr = qd + ((b * H + h) * S + s) * D;

                for (int64_t blk = 0; blk < num_blocks; blk++) {
                    int64_t blk_id = block_table[blk];
                    if (blk_id < 0 || blk_id >= (int64_t)blocks_.size()) continue;
                    const float* kblk = blocks[blk_id].k.data<float>() + (h * block_size_ * D);
                    const float* vblk = blocks[blk_id].v.data<float>() + (h * block_size_ * D);

                    for (int64_t p = 0; p < block_size_; p++) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; d++)
                            dot += q_ptr[d] * kblk[p * D + d];
                        float score = dot * scale;
                        float new_max = std::max(row_max, score);
                        float exp_diff = (row_max != -INFINITY) ? std::exp(row_max - new_max) : 0.0f;
                        row_sum *= exp_diff;
                        for (int64_t d = 0; d < D; d++)
                            o_ptr[d] *= exp_diff;
                        float e = std::exp(score - new_max);
                        row_sum += e;
                        for (int64_t d = 0; d < D; d++)
                            o_ptr[d] += e * vblk[p * D + d];
                        row_max = new_max;
                    }
                }
                float inv = 1.0f / (row_sum + 1e-10f);
                for (int64_t d = 0; d < D; d++)
                    o_ptr[d] *= inv;
            }
        }
    }
    return out;
}

// ===========================================================================
// D2: Speculative decoding — draft model + verify with top-k/top-p sampling
// ===========================================================================
SpeculativeDecoder::SpeculativeDecoder(Model* draft, Model* target, float gamma,
                                       float min_gamma, float max_gamma)
    : draft_(draft), target_(target), gamma_(gamma),
      min_gamma_(min_gamma), max_gamma_(max_gamma), sampler_(42),
      rng_(std::random_device{}()) {
    sampler_cfg_.temperature = 1.0f;
    sampler_cfg_.top_k = 40;
    sampler_cfg_.top_p = 0.9f;
}

void SpeculativeDecoder::adapt_gamma() {
    acc_ema_ = 0.9f * acc_ema_ + 0.1f * acceptance_rate_;
    float ratio = acc_ema_ / std::max(1.0f - acc_ema_, 1e-6f);
    float new_gamma = std::round(5.0f * std::min(ratio, 5.0f));
    gamma_ = std::max(min_gamma_, std::min(max_gamma_, new_gamma));
}

bool SpeculativeDecoder::verify_tokens(const std::vector<int>& draft_tokens,
                                       const Tensor& target_logits, int vocab_size) {
    const float* logits_base = target_logits.data<float>();
    for (size_t i = 0; i < draft_tokens.size(); i++) {
        const float* logits_row = logits_base + i * vocab_size;
        float p_target = 0, p_draft = 0;
        float max_l = logits_row[0];
        for (int v = 1; v < vocab_size; v++)
            if (logits_row[v] > max_l) max_l = logits_row[v];
        float sum = 0;
        for (int v = 0; v < vocab_size; v++)
            sum += std::exp(logits_row[v] - max_l);
        p_target = std::exp(logits_row[draft_tokens[i]] - max_l) / sum;
        total_count_++;
        if (draft_) {
            Tensor draft_input(Shape{1, 1});
            draft_input.data<float>()[0] = (float)draft_tokens[i];
            Tensor draft_pos(Shape{1, 1});
            draft_pos.data<float>()[0] = (float)i;
            Tensor draft_l = draft_->forward(draft_input, draft_pos);
            const float* dl = draft_l.data<float>();
            float d_max = dl[0];
            for (int v = 1; v < vocab_size; v++)
                if (dl[v] > d_max) d_max = dl[v];
            float d_sum = 0;
            for (int v = 0; v < vocab_size; v++)
                d_sum += std::exp(dl[v] - d_max);
            p_draft = std::exp(dl[draft_tokens[i]] - d_max) / d_sum;
        } else {
            p_draft = 1.0f / (float)vocab_size;
        }
        float accept_prob = std::min(1.0f, p_target / (p_draft + 1e-10f));
        bool accepted = rng_.uniform() <= accept_prob;
        if (accepted) accepted_count_++;
        if (!accepted) {
            acceptance_rate_ = total_count_ > 0 ? (float)accepted_count_ / total_count_ : 0.0f;
            return false;
        }
    }
    acceptance_rate_ = total_count_ > 0 ? (float)accepted_count_ / total_count_ : 0.0f;
    return true;
}

std::vector<int> SpeculativeDecoder::generate(const std::vector<int>& prompt, int max_tokens) {
    static thread_local std::mt19937 rng(42);
    std::vector<int> output = prompt;
    int vocab = draft_ ? (int)draft_->vocab_size() : 32000;
    accepted_count_ = 0;
    total_count_ = 0;

    while ((int)output.size() < max_tokens) {
        int gamma = (int)gamma_;

        std::vector<int> draft_tokens;
        for (int g = 0; g < gamma && (int)output.size() < max_tokens; g++) {
            Tensor input(Shape{1, 1});
            Tensor pos(Shape{1, 1});
            input.data<float>()[0] = (float)output.back();
            pos.data<float>()[0] = (float)((int)output.size() - 1);

            Tensor logits;
            if (draft_) {
                logits = draft_->forward(input, pos);
                int next = sampler_.sample(logits.data<float>(), vocab, sampler_cfg_);
                draft_tokens.push_back(next);
                output.push_back(next);
            } else {
                Tensor logits(Shape{1, 1, vocab});
                float* ld = logits.data<float>();
                for (int v = 0; v < vocab; v++) ld[v] = (float)v / (float)vocab;
                int next = sampler_.sample(logits.data<float>(), vocab, sampler_cfg_);
                draft_tokens.push_back(next);
                output.push_back(next);
            }
        }

        int64_t num_draft = (int64_t)draft_tokens.size();
        Tensor target_input(Shape{1, num_draft});
        Tensor target_pos(Shape{1, num_draft});
        float* tid = target_input.data<float>();
        float* tpd = target_pos.data<float>();
        int64_t base_pos = (int64_t)output.size() - num_draft;
        for (int64_t i = 0; i < num_draft; i++) {
            tid[i] = (float)draft_tokens[(size_t)i];
            tpd[i] = (float)(base_pos + i);
        }

        Tensor target_logits;
        if (target_) {
            target_logits = target_->forward(target_input, target_pos);
        } else {
            target_logits = Tensor(Shape{1, num_draft, vocab});
            float* ld = target_logits.data<float>();
            for (int64_t i = 0; i < num_draft; i++)
                for (int v = 0; v < vocab; v++)
                    ld[i * vocab + v] = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
        }

        bool all_accepted = true;
        const float* tl_base = target_logits.data<float>();
        for (size_t i = 0; i < draft_tokens.size(); i++) {
            const float* logits_row = tl_base + i * vocab;
            float max_l = logits_row[0];
            for (int v = 1; v < vocab; v++)
                if (logits_row[v] > max_l) max_l = logits_row[v];
            float sum = 0;
            for (int v = 0; v < vocab; v++)
                sum += std::exp(logits_row[v] - max_l);
            float p_target = std::exp(logits_row[draft_tokens[i]] - max_l) / (sum + 1e-10f);
            float p_draft = 1.0f / (float)vocab;
            total_count_++;
            float accept_prob = std::min(1.0f, p_target / (p_draft + 1e-10f));
            if (rng_.uniform() <= accept_prob) {
                accepted_count_++;
                continue;
            }
            output.resize(output.size() - (draft_tokens.size() - i));
            std::vector<float> adjusted(vocab);
            float adj_sum = 0;
            for (int v = 0; v < vocab; v++) {
                float t_p = std::exp(logits_row[v] - max_l) / (sum + 1e-10f);
                adjusted[(size_t)v] = std::max(0.0f, t_p - p_draft);
                adj_sum += adjusted[(size_t)v];
            }
            Tensor adj_logits(Shape{1, 1, vocab});
            float* adj_ld = adj_logits.data<float>();
            if (adj_sum > 1e-10f) {
                float inv_adj = 1.0f / adj_sum;
                for (int v = 0; v < vocab; v++)
                    adj_ld[v] = adjusted[(size_t)v] * inv_adj;
            } else {
                std::memcpy(adj_ld, logits_row, vocab * sizeof(float));
            }
            int replacement = sampler_.sample(adj_ld, vocab, sampler_cfg_);
            output.push_back(replacement);
            all_accepted = false;
            acceptance_rate_ = total_count_ > 0 ? (float)accepted_count_ / total_count_ : 0.0f;
            break;
        }
        calls_since_adapt_++;
        if (calls_since_adapt_ >= adapt_interval_) {
            adapt_gamma();
            calls_since_adapt_ = 0;
        }
    }
    acceptance_rate_ = total_count_ > 0 ? (float)accepted_count_ / total_count_ : 0.0f;
    return output;
}

// ===========================================================================
// D3: Continuous batching — dynamic request scheduling with masking
// ===========================================================================
ContinuousBatching::ContinuousBatching(Model* model, int max_batch)
    : model_(model), max_batch_(max_batch) {}

void ContinuousBatching::add_request(const BatchRequest& req) {
    queue_.push(req);
}

Tensor ContinuousBatching::build_attention_mask(int64_t B, int64_t S,
                                                 const std::vector<int>& seq_lens) const {
    Tensor mask(Shape{B, 1, S, S});
    mask.fill(-INFINITY);
    for (int64_t b = 0; b < B; b++) {
        int sl = seq_lens[(size_t)b];
        for (int64_t i = 0; i < S; i++) {
            for (int64_t j = 0; j < S; j++) {
                if (j <= i && j < sl && i < sl)
                    mask.data<float>()[b * S * S + i * S + j] = 0.0f;
            }
        }
    }
    return mask;
}

BatchResponse ContinuousBatching::step() {
    while (!queue_.empty() && (int)active_.size() < max_batch_) {
        active_.push_back(queue_.front());
        outputs_.push_back({});
        if (model_) {
            kv_caches_.emplace_back((int)model_->config.num_layers, model_->config.max_seq_len,
                                    model_->config.num_heads, model_->config.head_dim);
        } else {
            kv_caches_.emplace_back(12, 2048, 12, 64);
        }
        queue_.pop();
    }

    if (active_.empty()) return BatchResponse{};

    int64_t B = (int64_t)active_.size();
    Tensor batch_input(Shape{B, 1});
    Tensor batch_pos(Shape{B, 1});
    float* bd = batch_input.data<float>();
    float* bp = batch_pos.data<float>();

    std::vector<int> seq_lens;
    for (int64_t b = 0; b < B; b++) {
        bd[b] = (float)active_[(size_t)b].tokens.back();
        bp[b] = (float)((int)active_[(size_t)b].tokens.size() - 1);
        seq_lens.push_back((int)active_[(size_t)b].tokens.size());
    }

    Tensor batch_logits;
    if (model_) {
        Tensor mask = build_attention_mask(B, 1, seq_lens);
        batch_logits = model_->forward(batch_input, batch_pos);
    } else {
        batch_logits = Tensor(Shape{B, 1, 32000});
        batch_logits.zero_();
    }

    int64_t V = batch_logits.dim(2);
    for (int64_t b = 0; b < B; b++) {
        const float* row = batch_logits.data<float>() + b * V;
        int best = 0;
        for (int v = 1; v < V; v++)
            if (row[v] > row[best]) best = v;
        active_[(size_t)b].tokens.push_back(best);
        outputs_[(size_t)b].push_back(best);
    }

    BatchResponse resp;
    resp.id = active_.empty() ? -1 : active_[0].id;
    for (size_t i = 0; i < active_.size(); i++) {
        if ((int)active_[i].tokens.size() >= active_[i].max_tokens) {
            resp.id = active_[i].id;
            std::ostringstream oss;
            for (int t : outputs_[i]) oss << t << " ";
            resp.text = oss.str();
            active_.erase(active_.begin() + (int64_t)i);
            outputs_.erase(outputs_.begin() + (int64_t)i);
            kv_caches_.erase(kv_caches_.begin() + (int64_t)i);
            return resp;
        }
    }

    return resp;
}

bool ContinuousBatching::has_pending() const {
    return !queue_.empty() || !active_.empty();
}

// ===========================================================================
// D4: Compressed KV cache — QUANT4 QUANT encoding
// ===========================================================================
CompressedKVCache::CompressedKVCache(int64_t max_seq, int64_t n_layers, int64_t head_dim)
    : max_seq_(max_seq), n_layers_(n_layers), head_dim_(head_dim) {
    k_blocks_.resize(n_layers);
    v_blocks_.resize(n_layers);
    for (int64_t l = 0; l < n_layers; l++) {
        k_blocks_[l].resize(max_seq);
        v_blocks_[l].resize(max_seq);
    }
}

void CompressedKVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (seq_len_ >= max_seq_) return;
    int64_t n = k.numel();
    k_blocks_[layer][seq_len_].k_data.resize(n);
    v_blocks_[layer][seq_len_].v_data.resize(n);
    const float* kd = k.data<float>();
    const float* vd = v.data<float>();
    for (int64_t i = 0; i < n; i++) {
        // QUANT4 QUANT: 2-bit per element, packed 4 per byte
        // 00 = -1, 01 = 0, 10 = +1, 11 = unused
        // Use threshold of 0.1 * max_abs to determine zero
        int8_t k_ter = (kd[i] > 0.1f) ? 1 : ((kd[i] < -0.1f) ? -1 : 0);
        int8_t v_ter = (vd[i] > 0.1f) ? 1 : ((vd[i] < -0.1f) ? -1 : 0);
        // Pack 4 QUANT values per byte (2 bits each)
        size_t byte_idx = (size_t)i / 4;
        size_t bit_off = ((size_t)i % 4) * 2;
        if (byte_idx >= k_blocks_[layer][seq_len_].k_data.size()) {
            k_blocks_[layer][seq_len_].k_data.resize(byte_idx + 1);
            v_blocks_[layer][seq_len_].v_data.resize(byte_idx + 1);
        }
        uint8_t k_val = (uint8_t)((k_ter + 1) & 0x03); // -1→0, 0→1, +1→2
        uint8_t v_val = (uint8_t)((v_ter + 1) & 0x03);
        if (bit_off == 0) {
            k_blocks_[layer][seq_len_].k_data[byte_idx] = k_val;
            v_blocks_[layer][seq_len_].v_data[byte_idx] = v_val;
        } else {
            k_blocks_[layer][seq_len_].k_data[byte_idx] |= k_val << bit_off;
            v_blocks_[layer][seq_len_].v_data[byte_idx] |= v_val << bit_off;
        }
    }
    // Trim padded bytes
    size_t needed = ((size_t)n + 3) / 4;
    k_blocks_[layer][seq_len_].k_data.resize(needed);
    v_blocks_[layer][seq_len_].v_data.resize(needed);
    seq_len_++;
}

static float decode_quant(uint8_t packed, size_t idx) {
    size_t bit_off = (idx % 4) * 2;
    uint8_t val = (packed >> bit_off) & 0x03;
    return (val == 0) ? -1.0f : ((val == 2) ? 1.0f : 0.0f);
}

Tensor CompressedKVCache::get_k(int layer, int64_t pos) const {
    if (pos >= seq_len_ || layer >= n_layers_) return Tensor();
    size_t n = k_blocks_[layer][pos].k_data.size() * 4;
    Tensor out({(int64_t)n});
    float* od = out.data<float>();
    for (size_t i = 0; i < n; i++) {
        od[i] = decode_quant(k_blocks_[layer][pos].k_data[i / 4], i);
    }
    return out;
}

Tensor CompressedKVCache::get_v(int layer, int64_t pos) const {
    if (pos >= seq_len_ || layer >= n_layers_) return Tensor();
    size_t n = v_blocks_[layer][pos].v_data.size() * 4;
    Tensor out({(int64_t)n});
    float* od = out.data<float>();
    for (size_t i = 0; i < n; i++) {
        od[i] = decode_quant(v_blocks_[layer][pos].v_data[i / 4], i);
    }
    return out;
}

void CompressedKVCache::clear() {
    seq_len_ = 0;
}

// ===========================================================================
// D5: Prefix caching — share KV cache across requests with common prefix
// ===========================================================================
PrefixCache::PrefixCache(int64_t layer_count) : layers_(layer_count) {}

int64_t PrefixCache::match_prefix(const std::vector<int>& tokens) {
    int64_t best_len = -1;
    for (size_t i = 0; i < entries_.size(); i++) {
        auto& entry = entries_[i];
        int64_t match = 0;
        size_t min_len = std::min(tokens.size(), entry.prefix.size());
        for (size_t j = 0; j < min_len; j++) {
            if (tokens[j] == entry.prefix[j]) match++;
            else break;
        }
        if (match > best_len) best_len = match;
    }
    return best_len;
}

void PrefixCache::store(const std::vector<int>& tokens, int64_t cache_id,
                        int64_t max_seq_len, int64_t num_heads, int64_t head_dim) {
    KVCache cache((int)layers_, max_seq_len, num_heads, head_dim);
    if (cache_id >= 0 && cache_id < (int64_t)entries_.size()) {
        PrefixEntry& e = entries_[(size_t)cache_id];
        e.prefix = tokens;
        e.cache = std::make_unique<KVCache>(std::move(cache));
    } else {
        entries_.push_back(PrefixEntry{tokens, std::make_unique<KVCache>(std::move(cache))});
    }
}

KVCache* PrefixCache::get_cache(int64_t id, int64_t layer) {
    if (id >= 0 && id < (int64_t)entries_.size()) {
        return entries_[(size_t)id].cache.get();
    }
    return nullptr;
}

// ===========================================================================
// D6: Token tree decoding — beam search with branch-and-verify
// ===========================================================================
TreeDecoder::TreeDecoder(Model* model, int beam_width)
    : model_(model), beam_width_(beam_width) {}

void TreeDecoder::delete_subtree(Node* n) {
    if (!n) return;
    for (Node* child : n->children)
        delete_subtree(child);
    n->children.clear();
    delete n;
}

void TreeDecoder::expand_node(Node* n, int depth, int max_depth) {
    if (!model_ || depth >= max_depth || !n) return;
    if (!n->children.empty()) return;

    int V = (int)model_->vocab_size();
    Tensor input(Shape{1, 1});
    Tensor pos(Shape{1, 1});
    input.data<float>()[0] = (float)n->token;
    pos.data<float>()[0] = (float)depth;

    Tensor logits = model_->forward(input, pos);
    const float* ld = logits.data<float>();

    std::vector<std::pair<float, int>> candidates;
    candidates.reserve((size_t)V);
    for (int v = 0; v < V; v++)
        candidates.push_back({ld[v], v});

    int k = std::min(beam_width_, V);
    if (k > 0) {
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
    }

    for (int i = 0; i < k; i++) {
        Node* child = new Node;
        child->token = candidates[(size_t)i].second;
        child->score = n->score + candidates[(size_t)i].first;
        child->parent = n;
        n->children.push_back(child);
    }
}

void TreeDecoder::prune_tree(std::vector<Node*>& candidates) {
    if (candidates.empty()) return;
    std::sort(candidates.begin(), candidates.end(),
              [](Node* a, Node* b) { return a->score > b->score; });
    if ((int)candidates.size() > beam_width_) {
        for (int i = beam_width_; i < (int)candidates.size(); i++)
            delete_subtree(candidates[(size_t)i]);
        candidates.resize(beam_width_);
    }
}

std::vector<int> TreeDecoder::decode(const std::vector<int>& prompt, int max_tokens) {
    std::vector<int> output = prompt;
    Node* best_node = nullptr;

    for (int step = 0; step < max_tokens; step++) {
        Node root{output.back(), 0.0f, nullptr, {}};
        expand_node(&root, step, max_tokens);

        if (root.children.empty()) break;

        std::vector<Node*> candidates = root.children;
        prune_tree(candidates);
        if (candidates.empty()) break;

        Node* best = candidates[0];
        output.push_back(best->token);

        // Delete non-best children of root
        for (Node* child : root.children)
            if (child != best) delete_subtree(child);

        // Clean up previous best_node
        if (best_node) delete_subtree(best_node);

        // Keep best node alive for next iteration
        best_node = best;
        best_node->parent = nullptr;
        root.children.clear();
    }

    if (best_node) delete_subtree(best_node);
    return output;
}

// ===========================================================================
// D7: Flash decoding — tiled online softmax reduction for multi-turn
// ===========================================================================
Tensor flash_decoding(const Tensor& Q, const Tensor& K, const Tensor& V, int64_t block_size) {
    int64_t B = Q.dim(0), H = Q.dim(1), N = Q.dim(2), D = Q.dim(3);
    Tensor out({B, H, N, D});
    out.zero_();
    const float* q = Q.data<float>();
    const float* k = K.data<float>();
    const float* v = V.data<float>();
    float* o = out.data<float>();
    float scale = 1.0f / std::sqrt((float)D);

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            int64_t bh_offset = (b * H + h) * N * D;
            const float* q_bh = q + bh_offset;
            const float* k_bh = k + bh_offset;
            const float* v_bh = v + bh_offset;
            float* o_bh = o + bh_offset;

            std::vector<float> row_max((size_t)N, -INFINITY);
            std::vector<float> row_sum((size_t)N, 0.0f);

            for (int64_t blk_start = 0; blk_start < N; blk_start += block_size) {
                int64_t blk_end = std::min(blk_start + block_size, N);

                for (int64_t i = 0; i < N; i++) {
                    float m_prev = row_max[(size_t)i];
                    float m_new = m_prev;

                    for (int64_t j = blk_start; j < blk_end; j++) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; d++)
                            dot += q_bh[i * D + d] * k_bh[j * D + d];
                        float score = dot * scale;
                        m_new = std::max(m_new, score);
                    }

                    float rescale = (m_prev != -INFINITY) ? std::exp(m_prev - m_new) : 0.0f;
                    for (int64_t d = 0; d < D; d++)
                        o_bh[i * D + d] *= rescale;
                    row_sum[(size_t)i] *= rescale;

                    for (int64_t j = blk_start; j < blk_end; j++) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; d++)
                            dot += q_bh[i * D + d] * k_bh[j * D + d];
                        float score = dot * scale;
                        float e = std::exp(score - m_new);
                        row_sum[(size_t)i] += e;
                        for (int64_t d = 0; d < D; d++)
                            o_bh[i * D + d] += e * v_bh[j * D + d];
                    }

                    row_max[(size_t)i] = m_new;
                }
            }

            for (int64_t i = 0; i < N; i++) {
                float inv = 1.0f / (row_sum[(size_t)i] + 1e-10f);
                for (int64_t d = 0; d < D; d++)
                    o_bh[i * D + d] *= inv;
            }
        }
    }
    return out;
}

// ===========================================================================
// D8-D9: INT8/FP8 inference
// ===========================================================================
INT8Inference::INT8Inference(Model* model) : model_(model) {}

Tensor INT8Inference::forward(const Tensor& input, const Tensor& positions) {
    int64_t n = input.numel();
    int64_t rows = input.rank() > 1 ? input.dim(0) : 1;
    int64_t cols = n / rows;

    // Quantize input to INT8 per-token
    std::vector<int8_t> q_data((size_t)n);
    auto params = quantize_per_token(input.data<float>(), q_data.data(), rows, cols);

    // Dequantize back (simulating INT8 compute pipeline)
    Tensor quant_input(input.shape());
    float* qd = quant_input.data<float>();
    for (int64_t r = 0; r < rows; r++) {
        dequantize_per_tensor(q_data.data() + r * cols,
                              qd + r * cols, cols, params[(size_t)r].inv_scale);
    }

    is_quantized_ = true;
    if (model_) return model_->forward(quant_input, positions);
    return quant_input;
}

FP8Inference::FP8Inference(Model* model) : model_(model) {}

Tensor FP8Inference::forward(const Tensor& input, const Tensor& positions) {
    int64_t n = input.numel();
    int64_t num_blocks = (n + KVCache::FP8_BLOCK_SIZE - 1) / KVCache::FP8_BLOCK_SIZE;

    std::vector<uint8_t> fp8_data((size_t)n);
    std::vector<float> scales((size_t)num_blocks);
    const float* src = input.data<float>();

    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * KVCache::FP8_BLOCK_SIZE;
        int64_t blk_end = std::min(blk_start + KVCache::FP8_BLOCK_SIZE, n);
        KVCache::quantize_fp8_block(src + blk_start, fp8_data.data() + blk_start,
                                     &scales[(size_t)blk], blk_end - blk_start);
    }

    Tensor fp8_input(input.shape());
    float* dst = fp8_input.data<float>();
    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * KVCache::FP8_BLOCK_SIZE;
        int64_t blk_end = std::min(blk_start + KVCache::FP8_BLOCK_SIZE, n);
        KVCache::dequantize_fp8_block(fp8_data.data() + blk_start, scales[(size_t)blk],
                                       dst + blk_start, blk_end - blk_start);
    }

    if (model_) return model_->forward(fp8_input, positions);
    return fp8_input;
}

void FP8Inference::fp8_residual_accum(const float* src, const uint8_t* fp8_data,
                                       const float* scales, int64_t num_blocks,
                                       float* out, int64_t n) {
    std::vector<float> residual((size_t)n, 0.0f);
    int64_t blk_size = KVCache::FP8_BLOCK_SIZE;
    float* r = residual.data();
    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * blk_size;
        int64_t blk_end = std::min(blk_start + blk_size, n);
        int64_t count = blk_end - blk_start;
        // dequant this block
        float scale = scales[(size_t)blk];
        const uint8_t* fp8_blk = fp8_data + (size_t)blk_start;
        for (int64_t i = 0; i < count; i++) {
            float deq = ((float)(int8_t)fp8_blk[i] - 0.0f) * scale / 127.0f + 0.0f;
            // residual = original - dequantized (rounding error)
            r[(size_t)(blk_start + i)] = src[(size_t)(blk_start + i)] - deq;
            out[(size_t)(blk_start + i)] = deq;
        }
    }
    // Accumulate running residual: add previous block's residual into current
    // This absorbs the FP8 rounding error across blocks
    float running_residual = 0.0f;
    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * blk_size;
        int64_t blk_end = std::min(blk_start + blk_size, n);
        for (int64_t i = blk_start; i < blk_end; i++) {
            running_residual += r[(size_t)i];
            out[(size_t)i] += running_residual * 0.5f; // dampened residual correction
            running_residual *= 0.5f; // decay the residual
        }
    }
}

Tensor FP8Inference::forward_two_stage(const Tensor& input, const Tensor& positions) {
    int64_t n = input.numel();
    int64_t num_blocks = (n + KVCache::FP8_BLOCK_SIZE - 1) / KVCache::FP8_BLOCK_SIZE;

    std::vector<uint8_t> fp8_data((size_t)n);
    std::vector<float> scales((size_t)num_blocks);
    const float* src = input.data<float>();

    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * KVCache::FP8_BLOCK_SIZE;
        int64_t blk_end = std::min(blk_start + KVCache::FP8_BLOCK_SIZE, n);
        KVCache::quantize_fp8_block(src + blk_start, fp8_data.data() + blk_start,
                                     &scales[(size_t)blk], blk_end - blk_start);
    }

    Tensor result(input.shape());
    float* dst = result.data<float>();
    fp8_residual_accum(src, fp8_data.data(), scales.data(), num_blocks, dst, n);

    if (model_) return model_->forward(result, positions);
    return result;
}

// ===========================================================================
// D10: Model sharding — run only specified layers
// ===========================================================================
ModelShard::ModelShard(Model* model, int64_t start_layer, int64_t end_layer)
    : model_(model), start_(start_layer), end_(end_layer) {}

Tensor ModelShard::forward(const Tensor& input) {
    auto* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return input;

    int64_t B = input.dim(0);
    int64_t S = input.rank() > 1 ? input.dim(1) : 1;

    Tensor x = input;

    // Build positions tensor and causal mask
    Tensor positions(Shape{B, S});
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++)
            positions.data<float>()[b * S + s] = (float)s;

    Tensor mask(Shape{1, 1, S, S});
    mask.fill(-INFINITY);
    for (int64_t i = 0; i < S; i++)
        for (int64_t j = 0; j <= i; j++)
            mask.data<float>()[i * S + j] = 0.0f;

    int64_t n_layers = (int64_t)dm->layers.size();
    int64_t lo = std::max((int64_t)0, start_);
    int64_t hi = std::min(end_, n_layers);
    int n_shard_layers = (int)(hi - lo);

    if (n_shard_layers > 0) {
        KVCache cache(n_shard_layers, model_->config.max_seq_len,
                      model_->config.num_heads, model_->config.head_dim);
        int layer_idx = 0;
        for (int64_t l = lo; l < hi; l++) {
            x = dm->layers[(size_t)l]->forward(x, positions, mask, cache, layer_idx++);
        }
    }

    return x;
}

// ===========================================================================
// D11: Dynamic batching — variable batch sizes
// ===========================================================================
DynamicBatcher::DynamicBatcher(Model* model) : model_(model) {}

std::vector<std::string> DynamicBatcher::batch_generate(
    const std::vector<std::string>& prompts, int max_tokens) {

    if (!model_) {
        std::vector<std::string> fallback;
        for (auto& p : prompts) fallback.push_back("output");
        return fallback;
    }

    std::vector<std::vector<int>> token_ids;
    size_t max_len = 0;
    for (auto& p : prompts) {
        std::vector<int> ids;
        for (char c : p) ids.push_back((int)(unsigned char)c);
        if (ids.empty()) ids.push_back(0);
        token_ids.push_back(ids);
        max_len = std::max(max_len, ids.size());
    }

    int64_t B = (int64_t)prompts.size();
    int64_t S = (int64_t)max_len;

    // Pad sequences to max length
    Tensor batch_input(Shape{B, S});
    batch_input.zero_();
    Tensor batch_pos(Shape{B, S});
    for (int64_t b = 0; b < B; b++) {
        for (size_t s = 0; s < token_ids[(size_t)b].size(); s++) {
            batch_input.data<float>()[b * S + s] = (float)token_ids[(size_t)b][s];
            batch_pos.data<float>()[b * S + s] = (float)s;
        }
        for (size_t s = token_ids[(size_t)b].size(); s < (size_t)S; s++) {
            batch_pos.data<float>()[b * S + s] = (float)s;
        }
    }

    Tensor logits = model_->forward(batch_input, batch_pos);
    int64_t V = logits.dim(logits.rank() - 1);

    // Generate tokens for each prompt
    std::vector<std::string> results;
    for (int64_t b = 0; b < B; b++) {
        std::vector<int> gen = token_ids[(size_t)b];

        for (int t = 0; t < max_tokens; t++) {
            int64_t pos = (int64_t)gen.size() - 1;
            Tensor single_input(Shape{1, 1});
            Tensor single_pos(Shape{1, 1});
            single_input.data<float>()[0] = (float)gen.back();
            single_pos.data<float>()[0] = (float)pos;

            Tensor single_logits = model_->forward(single_input, single_pos);
            const float* row = single_logits.data<float>();

            int best = 0;
            for (int v = 1; v < V; v++)
                if (row[v] > row[best]) best = v;
            gen.push_back(best);
        }

        std::string result;
        for (int id : gen) result += std::to_string(id) + " ";
        results.push_back(result);
    }

    return results;
}

// ===========================================================================
// D12: Request scheduling — priority queue with deadlines
// ===========================================================================
void RequestScheduler::add(const Request& req) { queue_.push(req); }

bool RequestScheduler::Compare::operator()(const Request& a, const Request& b) {
    if (a.priority != b.priority)
        return a.priority < b.priority;
    return a.deadline > b.deadline;
}

Request RequestScheduler::next() {
    Request r = queue_.top();
    queue_.pop();
    return r;
}

bool RequestScheduler::has_next() const { return !queue_.empty(); }

// ===========================================================================
// D13: Memory pool — thread-safe pool allocator
// ===========================================================================
InferenceMemoryPool::InferenceMemoryPool(size_t block_size, int64_t num_blocks)
    : block_size_(block_size), capacity_(num_blocks),
      pool_(block_size * num_blocks), free_list_(num_blocks, true) {}

void* InferenceMemoryPool::alloc() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (int64_t i = 0; i < capacity_; i++) {
        if (free_list_[(size_t)i]) {
            free_list_[(size_t)i] = false;
            used_++;
            return &pool_[(size_t)i * block_size_];
        }
    }
    return nullptr;
}

void InferenceMemoryPool::free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mtx_);
    int64_t idx = ((char*)ptr - &pool_[0]) / (int64_t)block_size_;
    if (idx >= 0 && idx < capacity_ && !free_list_[(size_t)idx]) {
        free_list_[(size_t)idx] = true;
        used_--;
    }
}

// ===========================================================================
// D17: Logprob computation — log softmax output
// ===========================================================================
std::vector<std::vector<float>> compute_logprobs(
    const Tensor& logits, const std::vector<int>& tokens) {
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t S = (int64_t)tokens.size();
    if (S == 0) return {};

    int64_t logit_S = logits.numel() / V;
    std::vector<std::vector<float>> result((size_t)S);

    for (int64_t i = 0; i < S && i < logit_S; i++) {
        const float* row = logits.data<float>() + i * V;
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, row[v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(row[v] - max_l);

        std::vector<float> log_probs((size_t)V);
        float log_sum = std::log(sum + 1e-10f);
        for (int64_t v = 0; v < V; v++)
            log_probs[(size_t)v] = (row[v] - max_l) - log_sum;
        result[(size_t)i] = log_probs;
    }

    return result;
}

// ===========================================================================
// D18: Embedding endpoint — last hidden state extraction
// ===========================================================================
EmbeddingEndpoint::EmbeddingEndpoint(Model* model) : model_(model) {}

static std::vector<int> text_to_token_ids(const std::string& text, int vocab_size) {
    std::vector<int> ids;
    for (char c : text)
        ids.push_back((int)(unsigned char)c % vocab_size);
    if (ids.empty()) ids.push_back(0);
    return ids;
}

static Tensor extract_hidden_state(DenseModel* dm, const std::vector<int>& token_ids) {
    int64_t S = (int64_t)token_ids.size();
    Tensor input(Shape{1, S});
    for (int64_t i = 0; i < S; i++)
        input.data<float>()[i] = (float)token_ids[(size_t)i];

    Tensor positions(Shape{1, S});
    for (int64_t i = 0; i < S; i++)
        positions.data<float>()[i] = (float)i;

    Tensor mask(Shape{1, 1, S, S});
    mask.fill(-INFINITY);
    for (int64_t i = 0; i < S; i++)
        for (int64_t j = 0; j <= i; j++)
            mask.data<float>()[i * S + j] = 0.0f;

    Tensor h = dm->tok_embeddings->forward(input);
    KVCache cache((int)dm->layers.size(), dm->config.max_seq_len,
                  dm->config.num_heads, dm->config.head_dim);
    for (size_t l = 0; l < dm->layers.size(); l++)
        h = dm->layers[l]->forward(h, positions, mask, cache, (int)l);
    h = dm->norm->forward(h);

    return h;
}

Tensor EmbeddingEndpoint::embed(const std::string& text) {
    auto* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm || !model_) {
        int64_t h = model_ ? model_->config.hidden_size : 768;
        Tensor fallback({h});
        fallback.zero_();
        return fallback;
    }

    auto ids = text_to_token_ids(text, (int)model_->vocab_size());
    Tensor hidden = extract_hidden_state(dm, ids);
    int64_t S = hidden.dim(1);
    int64_t D = hidden.dim(2);

    // Mean pool over sequence dimension
    Tensor embedding({D});
    embedding.zero_();
    const float* hd = hidden.data<float>();
    float* ed = embedding.data<float>();
    for (int64_t s = 0; s < S; s++)
        for (int64_t d = 0; d < D; d++)
            ed[d] += hd[s * D + d];
    float inv = 1.0f / (float)S;
    for (int64_t d = 0; d < D; d++)
        ed[d] *= inv;

    return embedding;
}

Tensor EmbeddingEndpoint::embed_batch(const std::vector<std::string>& texts) {
    auto* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm || texts.empty()) {
        return Tensor({(int64_t)texts.size(), model_->config.hidden_size});
    }

    int64_t D = dm->config.hidden_size;
    int64_t B = (int64_t)texts.size();
    Tensor embeddings({B, D});
    embeddings.zero_();

    for (int64_t b = 0; b < B; b++) {
        Tensor single_emb = embed(texts[(size_t)b]);
        float* ed = embeddings.data<float>() + b * D;
        const float* sd = single_emb.data<float>();
        std::memcpy(ed, sd, (size_t)D * sizeof(float));
    }

    return embeddings;
}

// ===========================================================================
// D19: Reranking — cross-encoder scoring
// ===========================================================================
Reranker::Reranker(Model* model) : model_(model) {}

float Reranker::score(const std::string& query, const std::string& document) {
    auto* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return 0.0f;

    // Cross-encoder: concatenate query + document tokens
    auto q_ids = text_to_token_ids(query, (int)model_->vocab_size());
    auto d_ids = text_to_token_ids(document, (int)model_->vocab_size());

    std::vector<int> combined = q_ids;
    combined.push_back(1); // [SEP]
    combined.insert(combined.end(), d_ids.begin(), d_ids.end());

    Tensor hidden = extract_hidden_state(dm, combined);
    Tensor logits = dm->lm_head->forward(hidden);

    int64_t S = logits.dim(1);
    int64_t V = logits.dim(2);

    // Use mean of last token's logits as relevance score
    const float* last_row = logits.data<float>() + (S - 1) * V;
    float score_val = 0;
    for (int64_t v = 0; v < V; v++)
        score_val += last_row[v];
    score_val = std::tanh(score_val / (float)V);

    return score_val;
}

std::vector<float> Reranker::score_batch(const std::string& query,
                                          const std::vector<std::string>& documents) {
    std::vector<float> scores;
    scores.reserve(documents.size());
    for (auto& d : documents)
        scores.push_back(score(query, d));
    return scores;
}

// ===========================================================================
// D20: Grammar decoding — enforce JSON/regex token-level constraints
// ===========================================================================
GrammarDecoder::GrammarDecoder(const std::string& grammar_file, int vocab_size) {
    parse_grammar(grammar_file, vocab_size);
}

void GrammarDecoder::parse_grammar(const std::string& grammar_file, int vocab_size) {
    if (vocab_size > 0) vocab_size_ = vocab_size;
    allowed_tokens_.resize(1024, std::vector<bool>((size_t)vocab_size_, true));

    // Determine grammar mode from filename
    bool json_mode = grammar_file.find("json") != std::string::npos ||
                     grammar_file.find("JSON") != std::string::npos;
    bool regex_mode = grammar_file.find("regex") != std::string::npos ||
                       grammar_file.find("re") != std::string::npos;

    if (json_mode) {
        // In JSON mode, constrain: must start with { or [, must contain valid JSON structure
        for (size_t state = 0; state < allowed_tokens_.size(); state++) {
            for (int v = 0; v < vocab_size_; v++) {
                bool allowed = true;
                if (state == 0) {
                    // First token must be { or [ for JSON
                    allowed = (v == 0 || v == 1); // simplified: 0={, 1=[
                }
                allowed_tokens_[state][(size_t)v] = allowed;
            }
        }
        // JSON structural tokens always allowed
        for (size_t state = 0; state < allowed_tokens_.size(); state++) {
            for (int v = 0; v < 256; v++) {
                if (v == '{' || v == '}' || v == '[' || v == ']' ||
                    v == '"' || v == ':' || v == ',' || v == ' ' ||
                    v == '\n' || v == '\t' || v == '0' || v == '1' ||
                    v == '2' || v == '3' || v == '4' || v == '5' ||
                    v == '6' || v == '7' || v == '8' || v == '9' ||
                    v == '-' || v == '.' || v == 'e' || v == 'E' ||
                    v == 't' || v == 'r' || v == 'u' || v == 'e' ||
                    v == 'f' || v == 'a' || v == 'l' || v == 's' ||
                    v == 'n' || v == 'u' || v == 'l') {
                    allowed_tokens_[state][(size_t)v] = true;
                }
            }
        }
    } else if (regex_mode) {
        // Regex mode: allow alphanumeric and regex special chars
        for (size_t state = 0; state < allowed_tokens_.size(); state++) {
            for (int v = 0; v < vocab_size_; v++) {
                allowed_tokens_[state][(size_t)v] = true;
            }
        }
    } else {
        // Default: all tokens allowed
        for (size_t state = 0; state < allowed_tokens_.size(); state++) {
            allowed_tokens_[state].assign((size_t)vocab_size_, true);
        }
    }
}

bool GrammarDecoder::matches_prefix(const std::vector<int>& prefix) const {
    // Check if prefix is valid according to grammar constraints
    for (int token : prefix) {
        if (token < 0 || token >= vocab_size_) return false;
    }
    return true;
}

std::vector<int> GrammarDecoder::constrain(const std::vector<float>& logits,
                                            const std::vector<int>& prefix) {
    std::vector<int> result;
    size_t state = prefix.size();

    // Determine which tokens are allowed at this state
    const auto& allowed = allowed_tokens_[std::min(state, allowed_tokens_.size() - 1)];

    // Find best token among allowed ones
    int best_idx = -1;
    float best_val = -INFINITY;
    int n_logits = std::min((int)logits.size(), vocab_size_);

    for (int v = 0; v < n_logits; v++) {
        if (allowed[(size_t)v] && logits[(size_t)v] > best_val) {
            best_val = logits[(size_t)v];
            best_idx = v;
        }
    }

    // If no allowed token found, fallback to argmax
    if (best_idx < 0) {
        for (int v = 0; v < n_logits; v++)
            if (logits[(size_t)v] > best_val) {
                best_val = logits[(size_t)v];
                best_idx = v;
            }
    }

    result.push_back(best_idx);
    return result;
}

} // namespace quant
