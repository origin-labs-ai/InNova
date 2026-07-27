#define NOMINMAX
#include "oil/trainer.h"
#include "oil/math.h"
#include "oil/autograd.h"
#include "oil/optimizer.h"
#include "oil/transformer.h"
#include "oil/flash_attention.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace oil {

PPOTrainer::PPOTrainer(Model* policy, Model* ref_model, float clip_epsilon,
                       float value_coef, float entropy_coef,
                       float gamma, float gae_lambda, float kl_target)
    : policy_(policy), ref_model_(ref_model),
      clip_epsilon_(clip_epsilon), value_coef_(value_coef),
      entropy_coef_(entropy_coef), gamma_(gamma),
      gae_lambda_(gae_lambda), kl_target_(kl_target), kl_alpha_(0.0f) {
    int64_t hs = policy_->config.hidden_size;
    v_fc1_weight_ = Tensor(Shape{hs, 64});
    v_fc1_bias_ = Tensor(Shape{64});
    v_fc2_weight_ = Tensor(Shape{64, 1});
    v_fc2_bias_ = Tensor(Shape{1});
    static thread_local std::mt19937 rng(42);
    float scale = 1.0f / std::sqrt((float)hs);
    float* ptr = v_fc1_weight_.data<float>();
    for (int64_t i = 0; i < v_fc1_weight_.numel(); i++)
        ptr[i] = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * scale;
    ptr = v_fc1_bias_.data<float>();
    for (int64_t i = 0; i < v_fc1_bias_.numel(); i++) ptr[i] = 0.0f;
    ptr = v_fc2_weight_.data<float>();
    for (int64_t i = 0; i < v_fc2_weight_.numel(); i++)
        ptr[i] = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * 0.1f;
    ptr = v_fc2_bias_.data<float>();
    ptr[0] = 0.0f;
}

Tensor PPOTrainer::critic_forward(const Tensor& hidden) {
    int64_t B = hidden.dim(0);
    int64_t D = hidden.dim(1);

    Tensor h1(Shape{B, 64});
    kernel::scalar_gemm(hidden.data<float>(), v_fc1_weight_.data<float>(),
                        h1.data<float>(), (int)B, 64, (int)D);
    const float* b1 = v_fc1_bias_.data<float>();
    float* h1d = h1.data<float>();
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < 64; j++)
            h1d[i * 64 + j] += b1[j];
    for (int64_t i = 0; i < B * 64; i++)
        h1d[i] = h1d[i] > 0.0f ? h1d[i] : 0.0f;

    Tensor out(Shape{B, 1});
    kernel::scalar_gemm(h1.data<float>(), v_fc2_weight_.data<float>(),
                        out.data<float>(), (int)B, 1, 64);
    const float* b2 = v_fc2_bias_.data<float>();
    float* od = out.data<float>();
    for (int64_t i = 0; i < B; i++) od[i] += b2[0];
    return out;
}

Tensor PPOTrainer::compute_log_probs(const Tensor& logits, const Tensor& ids) {
    int64_t B = ids.dim(0);
    int64_t S = ids.dim(1);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();
    const float* id = ids.data<float>();
    Tensor logprobs(Shape{B, S});
    float* lpd = logprobs.data<float>();
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = i * S + j;
            int64_t target = (int64_t)id[idx];
            if (target < 0) target = 0;
            if (target >= V) target = V - 1;
            const float* row = lp + idx * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++)
                if (row[v] > max_l) max_l = row[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++)
                sum_exp += std::exp(row[v] - max_l);
            float log_prob = row[target] - max_l - std::log(sum_exp + 1e-10f);
            lpd[idx] = log_prob;
        }
    }
    return logprobs;
}

Tensor PPOTrainer::compute_gae(const Tensor& rewards, const Tensor& values,
                                const Tensor& dones) {
    int64_t T = rewards.dim(0);
    const float* rd = rewards.data<float>();
    const float* vd = values.data<float>();
    const float* dd = dones.data<float>();
    Tensor advantages(Shape{T});
    Tensor returns(Shape{T});
    float* ad = advantages.data<float>();
    float* retd = returns.data<float>();
    float gae = 0.0f;
    float next_val = 0.0f;
    for (int64_t t = T - 1; t >= 0; t--) {
        float done_mask = 1.0f - dd[t];
        float delta = rd[t] + gamma_ * next_val * done_mask - vd[t];
        gae = delta + gamma_ * gae_lambda_ * done_mask * gae;
        ad[t] = gae;
        retd[t] = ad[t] + vd[t];
        next_val = vd[t];
    }
    return advantages;
}

float PPOTrainer::compute_kl_divergence(const Tensor& logits_a, const Tensor& logits_b) {
    int64_t N = logits_a.dim(0);
    int64_t V = logits_a.dim(logits_a.rank() - 1);
    const float* la = logits_a.data<float>();
    const float* lb = logits_b.data<float>();
    float kl = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        float max_a = -INFINITY, max_b = -INFINITY;
        for (int64_t v = 0; v < V; v++) {
            if (la[i * V + v] > max_a) max_a = la[i * V + v];
            if (lb[i * V + v] > max_b) max_b = lb[i * V + v];
        }
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int64_t v = 0; v < V; v++) {
            sum_a += std::exp(la[i * V + v] - max_a);
            sum_b += std::exp(lb[i * V + v] - max_b);
        }
        float inv_a = 1.0f / (sum_a + 1e-10f);
        float inv_b = 1.0f / (sum_b + 1e-10f);
        for (int64_t v = 0; v < V; v++) {
            float pa = std::exp(la[i * V + v] - max_a) * inv_a;
            float pb = std::exp(lb[i * V + v] - max_b) * inv_b;
            if (pa > 1e-10f)
                kl += pa * (std::log(pa + 1e-10f) - std::log(pb + 1e-10f));
        }
    }
    return kl / (float)N;
}

void PPOTrainer::train_step(const Tensor& states, const Tensor& actions,
                             const Tensor& old_logprobs, const Tensor& advantages,
                             const Tensor& returns) {
    int64_t B = states.dim(0);
    int64_t S = actions.dim(1);
    int64_t V = policy_->config.vocab_size;
    int64_t HS = policy_->config.hidden_size;

    Tensor positions(Shape{B, S});
    float* ps = positions.data<float>();
    for (int64_t i = 0; i < B * S; i++) ps[i] = (float)(i % S);

    Tensor logits = policy_->forward(states, positions, nullptr);
    Tensor ref_logits = ref_model_ ? ref_model_->forward(states, positions, nullptr) : logits;

    Tensor values = critic_forward(states);

    Tensor logprobs = compute_log_probs(logits, actions);
    const float* lp = logprobs.data<float>();
    const float* olp = old_logprobs.data<float>();
    const float* adv = advantages.data<float>();
    const float* ret = returns.data<float>();
    const float* vd = values.data<float>();

    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy = 0.0f;
    float kl_div = 0.0f;

    if (ref_model_) {
        Tensor ref_logprobs = compute_log_probs(ref_logits, actions);
        const float* rlp = ref_logprobs.data<float>();
        for (int64_t i = 0; i < B * S; i++)
            kl_div += std::exp(olp[i]) * (olp[i] - rlp[i]);
        kl_div /= (float)(B * S);
    }

    float adv_mean = 0.0f, adv_std = 0.0f;
    int64_t n_adv = advantages.numel();
    const float* ad = advantages.data<float>();
    for (int64_t i = 0; i < n_adv; i++) adv_mean += ad[i];
    adv_mean /= (float)n_adv;
    for (int64_t i = 0; i < n_adv; i++) adv_std += (ad[i] - adv_mean) * (ad[i] - adv_mean);
    adv_std = std::sqrt(adv_std / (float)n_adv + 1e-8f);

    for (int64_t i = 0; i < B * S; i++) {
        float ratio = std::exp(lp[i] - olp[i]);
        float clip_high = 1.0f + clip_epsilon_;
        float clip_low = 1.0f - clip_epsilon_;
        float clipped = std::min(std::max(ratio, clip_low), clip_high);
        float adv_norm = (adv[i % B] - adv_mean) / adv_std;
        policy_loss -= std::min(ratio * adv_norm, clipped * adv_norm);
    }
    policy_loss /= (float)(B * S);

    for (int64_t i = 0; i < B; i++) {
        float diff = vd[i] - ret[i];
        value_loss += diff * diff;
    }
    value_loss *= 0.5f / (float)B;

    const float* lpd = logits.data<float>();
    for (int64_t i = 0; i < B * S; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++)
            if (lpd[i * V + v] > max_l) max_l = lpd[i * V + v];
        float sum_exp = 0.0f;
        for (int64_t v = 0; v < V; v++)
            sum_exp += std::exp(lpd[i * V + v] - max_l);
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (int64_t v = 0; v < V; v++) {
            float p = std::exp(lpd[i * V + v] - max_l) * inv_sum;
            if (p > 1e-10f) entropy -= p * std::log(p);
        }
    }
    entropy /= (float)(B * S);

    float kl_scale = 1.0f;
    if (kl_div > kl_target_ * 2.0f) kl_scale = 0.0f;
    else if (kl_div > kl_target_ * 1.5f) kl_alpha_ += 0.002f;
    else if (kl_div < kl_target_ * 0.5f) kl_alpha_ = std::max(0.0f, kl_alpha_ - 0.001f);
    last_kl_ = kl_div;

    float total_loss = policy_loss + value_coef_ * value_loss - entropy_coef_ * entropy + kl_alpha_ * kl_div;

    if (log_cb_) log_cb_(policy_loss, value_loss, entropy, kl_div);

    for (auto* p : critic_parameters()) {
        if (p->has_grad()) p->zero_grad();
    }

    float* vl = values.data<float>();
    if (!v_fc2_bias_.has_grad()) {
        v_fc2_bias_.set_grad(Tensor(Shape{1}));
        v_fc2_bias_.grad().zero_();
    }
    if (!v_fc2_weight_.has_grad()) {
        v_fc2_weight_.set_grad(Tensor(v_fc2_weight_.shape()));
        v_fc2_weight_.grad().zero_();
    }
    if (!v_fc1_bias_.has_grad()) {
        v_fc1_bias_.set_grad(Tensor(v_fc1_bias_.shape()));
        v_fc1_bias_.grad().zero_();
    }
    if (!v_fc1_weight_.has_grad()) {
        v_fc1_weight_.set_grad(Tensor(v_fc1_weight_.shape()));
        v_fc1_weight_.grad().zero_();
    }

    for (int64_t i = 0; i < B; i++) {
        float diff = (vl[i] - ret[i]);
        v_fc2_bias_.grad().data<float>()[0] += diff;
        for (int64_t j = 0; j < 64; j++) {
            v_fc2_weight_.grad().data<float>()[j] += diff * states.data<float>()[i * states.dim(1) + j];
        }
    }

    for (int64_t i = 0; i < B; i++) {
        float diff = (vl[i] - ret[i]);
        for (int64_t j = 0; j < 64; j++) {
            float grad_fc2 = diff * v_fc2_weight_.data<float>()[j];
            float hs_val = states.data<float>()[i * states.dim(1) + j];
            for (int64_t k = 0; k < HS; k++) {
                float relu_gate = hs_val * (hs_val > 0.0f ? 1.0f : 0.0f);
                v_fc1_weight_.grad().data<float>()[k * 64 + j] += grad_fc2 * relu_gate;
            }
            if (hs_val > 0.0f)
                v_fc1_bias_.grad().data<float>()[j] += grad_fc2;
        }
    }

    float v_scale = value_coef_ / (float)B;
    for (int64_t i = 0; i < v_fc1_weight_.numel(); i++)
        v_fc1_weight_.grad().data<float>()[i] *= v_scale;
    for (int64_t i = 0; i < v_fc1_bias_.numel(); i++)
        v_fc1_bias_.grad().data<float>()[i] *= v_scale;
    v_fc2_weight_.grad().data<float>()[0] *= value_coef_ / (float)B;
    v_fc2_bias_.grad().data<float>()[0] *= value_coef_ / (float)B;
}

std::vector<Tensor*> PPOTrainer::critic_parameters() {
    return {&v_fc1_weight_, &v_fc1_bias_, &v_fc2_weight_, &v_fc2_bias_};
}

DPOTrainer::DPOTrainer(Model* policy, Model* ref_model, float beta)
    : policy_(policy), ref_model_(ref_model), beta_(beta) {}

float DPOTrainer::compute_log_probs(const Tensor& logits, const Tensor& ids, Tensor* out_probs) {
    int64_t B = ids.dim(0);
    int64_t S = ids.dim(1);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();
    const float* id = ids.data<float>();

    if (out_probs) *out_probs = Tensor(Shape{B});
    float* op = out_probs ? out_probs->data<float>() : nullptr;

    float total_logprob = 0.0f;
    for (int64_t i = 0; i < B; i++) {
        float seq_logprob = 0.0f;
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = i * S + j;
            int64_t target = (int64_t)id[idx];
            if (target < 0) target = 0;
            if (target >= V) target = V - 1;
            const float* row = lp + idx * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++)
                if (row[v] > max_l) max_l = row[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++)
                sum_exp += std::exp(row[v] - max_l);
            float log_prob = row[target] - max_l - std::log(sum_exp + 1e-10f);
            seq_logprob += log_prob;
        }
        if (op) op[i] = seq_logprob;
        total_logprob += seq_logprob;
    }
    return total_logprob / (float)B;
}

float DPOTrainer::train_step(const Tensor& chosen_logits, const Tensor& rejected_logits,
                              const Tensor& chosen_ids, const Tensor& rejected_ids) {
    Tensor log_pi_chosen, log_pi_rejected;
    Tensor log_ref_chosen, log_ref_rejected;
    int64_t B = chosen_ids.dim(0);

    compute_log_probs(chosen_logits, chosen_ids, &log_pi_chosen);
    compute_log_probs(rejected_logits, rejected_ids, &log_pi_rejected);

    if (ref_model_) {
        int64_t HS = policy_->config.hidden_size;
        int64_t S = chosen_ids.dim(1);
        Tensor positions(Shape{B, S});
        float* ps = positions.data<float>();
        for (int64_t i = 0; i < B * S; i++) ps[i] = (float)(i % S);

        Tensor ref_chosen = ref_model_->forward(chosen_ids, positions, nullptr);
        Tensor ref_rejected = ref_model_->forward(rejected_ids, positions, nullptr);
        compute_log_probs(ref_chosen, chosen_ids, &log_ref_chosen);
        compute_log_probs(ref_rejected, rejected_ids, &log_ref_rejected);
    } else {
        log_ref_chosen = Tensor(Shape{B});
        log_ref_rejected = Tensor(Shape{B});
        log_ref_chosen.zero_();
        log_ref_rejected.zero_();
    }

    const float* lpc = log_pi_chosen.data<float>();
    const float* lpr = log_pi_rejected.data<float>();
    const float* lrc = log_ref_chosen.data<float>();
    const float* lrr = log_ref_rejected.data<float>();

    float loss_val = 0.0f;
    float kl_val = 0.0f;
    for (int64_t i = 0; i < B; i++) {
        float reward_chosen = lpc[i] - lrc[i];
        float reward_rejected = lpr[i] - lrr[i];
        float diff = beta_ * (reward_chosen - reward_rejected);
        float sig = 1.0f / (1.0f + std::exp(-diff));
        loss_val -= std::log(std::max(sig, 1e-10f));
        kl_val += (lrc[i] - lpc[i]) + (lpr[i] - lrr[i]);
    }
    loss_val /= (float)B;
    kl_val /= (float)(B * 2);
    last_loss_ = loss_val;

    if (log_cb_) log_cb_(loss_val, kl_val);

    return loss_val;
}

RLHFPipeline::RLHFPipeline(Model* model, Model* ref_model, Tokenizer* tokenizer,
                           RewardModel* reward_model, Trainer* trainer,
                           Optimizer* policy_opt, Optimizer* rm_opt)
    : model_(model), ref_model_(ref_model), tokenizer_(tokenizer),
      reward_model_(reward_model), trainer_(trainer),
      policy_opt_(policy_opt), rm_opt_(rm_opt) {}

Tensor RLHFPipeline::extract_hidden(Tensor& logits, int64_t hidden_size) {
    int64_t B = logits.dim(0);
    Tensor pooled(Shape{B, hidden_size});
    float* pd = pooled.data<float>();
    const float* ld = logits.data<float>();
    int64_t total = logits.numel();
    int64_t feats = total / B;
    int64_t pool_dim = std::min(feats, hidden_size);
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < pool_dim; j++)
            pd[i * hidden_size + j] = ld[i * feats + (feats - pool_dim) + j];
        for (int64_t j = pool_dim; j < hidden_size; j++)
            pd[i * hidden_size + j] = 0.0f;
    }
    return pooled;
}

Tensor RLHFPipeline::get_reward_for_sequence(Model* model, const std::vector<int>& ids) {
    int64_t S = (int64_t)ids.size();
    Tensor input_ids(Shape{1, S});
    Tensor positions(Shape{1, S});
    float* idp = input_ids.data<float>();
    float* psp = positions.data<float>();
    for (int64_t i = 0; i < S; i++) {
        idp[i] = (float)ids[i];
        psp[i] = (float)i;
    }
    Tensor logits = model->forward(input_ids, positions, nullptr);
    int64_t hidden_size = model->config.hidden_size;
    if (hidden_size <= 0) hidden_size = 64;
    Tensor feat = extract_hidden(logits, hidden_size);
    return reward_model_->score(feat);
}

void RLHFPipeline::generate_comparisons(const std::vector<std::string>& prompts, int max_new_tokens) {
    int vocab_size = (int)model_->config.vocab_size;

    for (size_t p = 0; p < prompts.size(); p++) {
        auto tokens = tokenizer_ ? tokenizer_->encode(prompts[p]) : std::vector<int>();
        if (tokens.empty()) {
            int offset = 5;
            int mod = std::max(1, vocab_size - offset);
            for (char c : prompts[p])
                tokens.push_back((int)(unsigned char)c % mod + offset);
        }

        std::vector<int> all_ids = tokens;
        int context = std::min((int)model_->config.max_seq_len, 512);

        for (int step = 0; step < max_new_tokens; step++) {
            int64_t len = (int64_t)all_ids.size();
            int64_t start = std::max((int64_t)0, len - context);
            int64_t ctx_len = len - start;

            Tensor input_ids(Shape{1, ctx_len});
            Tensor positions(Shape{1, ctx_len});
            float* idp = input_ids.data<float>();
            float* psp = positions.data<float>();
            for (int64_t i = 0; i < ctx_len; i++) {
                idp[i] = (float)all_ids[start + i];
                psp[i] = (float)(start + i);
            }
            Tensor logits = model_->forward(input_ids, positions, nullptr);
            int64_t V = logits.dim(logits.rank() - 1);
            const float* lp = logits.data<float>();
            const float* last_row = lp + (ctx_len - 1) * V;
            int next = 0;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) {
                if (last_row[v] > max_l) { max_l = last_row[v]; next = (int)v; }
            }
            all_ids.push_back(next);
            if (next < 2) break;
        }

        std::vector<int> generation(all_ids.begin() + (int64_t)tokens.size(), all_ids.end());

        Comparison comp;
        comp.prompt = prompts[p];
        comp.chosen_ids = all_ids;
        comp.rejected_ids = generation;

        Tensor chosen_rew = get_reward_for_sequence(model_, comp.chosen_ids);
        Tensor gen_rew = get_reward_for_sequence(model_, comp.rejected_ids);
        comp.reward_chosen = chosen_rew.data<float>()[0];
        comp.reward_rejected = gen_rew.data<float>()[0];

        if (comp.reward_chosen <= comp.reward_rejected) {
            std::swap(comp.chosen_ids, comp.rejected_ids);
            std::swap(comp.reward_chosen, comp.reward_rejected);
        }

        comparison_buffer_.push_back(comp);

        if (verbose_) {
            std::cout << "[RLHF] Comparison " << (p + 1) << "/" << prompts.size()
                      << " | reward_chosen=" << comp.reward_chosen
                      << " reward_rejected=" << comp.reward_rejected << std::endl;
        }
    }
}
} // namespace oil
