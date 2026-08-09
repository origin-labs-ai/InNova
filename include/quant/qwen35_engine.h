#pragma once
#include "quant/quant_format.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace quant {

class Qwen35Engine {
public:
    Qwen35Engine();
    ~Qwen35Engine();

    bool load(const std::string& quant_path);
    bool ok() const { return ok_; }

    void reset();                                    // clear all caches/states
    void prefill(const std::vector<int>& ids);       // batched forward, no sampling
    void append_token(int id);                       // single-token decode step
    int sample(float temperature, int top_k);        // sample from last logits
    size_t pos() const { return pos_; }
    int vocab() const { return vocab_; }
    const float* logits() const { return logits_.data(); }
    const std::vector<float>& last_h() const { return last_h_; }
    const float* trace(int L) const { return trace_.data() + (size_t)L * 4096; }

private:
    struct Th { uint32_t start; uint32_t count; Format fmt; uint32_t rows; uint32_t cols; };

    const Th* th(const std::string& n) const {
        auto it = th_.find(n);
        return it == th_.end() ? nullptr : &it->second;
    }
    bool resolve(const std::string& name, uint32_t rows, uint32_t cols);
    bool load_layer(int L);

    void gemm(const Th& t, const float* X, float* Y, int B);   // Y[B][rows] = X[B][cols] @ W^T
    void rmsnorm(float* out, const float* x, const float* w, int n);
    void gla_step(int L, const float* x, float* y);            // single token
    void attn_step(int L, const float* x, float* y, int pos);
    void mlp_step(int L, const float* x, float* y);
    void run_logits(const float* h);
    void gla_prefill(int L, const float* qkv_vec, const float* z_vec, const float* a_vec,
                     const float* b_vec, float* y);
    void attn_prefill(int L, const float* qvec, const float* kvec, const float* vvec,
                      int pos, float* y);

    QUANTReader* reader_;
    bool ok_;
    bool skip_full_ = false;   // debug: bypass full-attention layers
    bool skip_gla_ = false;    // debug: bypass GLA layers
    int vocab_;
    std::unordered_map<std::string, Th> th_;

    size_t pos_;                       // tokens seen (incl. prefill)
    std::vector<float> h_;             // 4096
    std::vector<float> logits_;        // vocab_
    std::vector<float> last_h_;        // last token post-final-norm (debug)
    std::vector<float> trace_;         // per-layer hidden traces (32*4096, debug)

    // persistent per-layer state
    std::vector<float> conv_state_;    // 24 * 8192 * 3
    std::vector<float> gla_state_;     // 24 * 32 * 128 * 128
    std::vector<float> kv_k_;          // 8 * 4 * max_kv * 256
    std::vector<float> kv_v_;          // 8 * 4 * max_kv * 256
    int max_kv_;

    // per-tensor small constants loaded at init
    std::vector<float> A_log_[32];
    std::vector<float> dt_bias_[32];
    std::vector<float> conv_w_[32];    // 8192*4
    std::vector<float> gnorm_w_[32];   // 128
    std::vector<float> qnorm_w_[32];   // 256
    std::vector<float> knorm_w_[32];   // 256
    std::vector<float> ln1_w_[32];     // 4096
    std::vector<float> ln2_w_[32];     // 4096
    std::vector<float> final_norm_w_;  // 4096

    // work buffers
    std::vector<float> qkv_;           // 8192
    std::vector<float> xb_, ob_;       // batch activation buffers
    std::vector<float> ga_, up_;       // mlp batch buffers
    std::vector<float> tmp_;
    float* scratch_;                   // 256 floats

    uint64_t rng_;

    static void decode_block(const QUANTReader& rd, uint32_t block_id, uint32_t nw, float* out);
};

} // namespace quant
