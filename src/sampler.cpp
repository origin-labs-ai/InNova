#include "quant/sampler.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <vector>

namespace quant {

Sampler::Sampler(uint64_t seed) : rng_(seed) {}

int Sampler::greedy(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

void Sampler::apply_temperature(float* logits, int n, float temp) const {
    if (temp < 1e-5f) temp = 1e-5f;
    for (int i = 0; i < n; i++) {
        logits[i] /= temp;
    }
}

void Sampler::apply_repetition_penalty(float* logits, int n,
                                        const std::vector<int>& prev_tokens,
                                        float penalty) const {
    if (penalty <= 1.0f) return;
    for (int t : prev_tokens) {
        if (t >= 0 && t < n) {
            logits[t] = (logits[t] < 0) ? logits[t] * penalty : logits[t] / penalty;
        }
    }
}

int Sampler::sample_top_k(const float* logits, int vocab_size, int k, float temp) {
    if (k <= 0 || k >= vocab_size) k = vocab_size;
    if (temp < 1e-5f) return greedy(logits, vocab_size);
    
    std::vector<std::pair<float, int>> scored(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        scored[i] = {logits[i] / temp, i};
    }
    
    std::nth_element(scored.begin(), scored.begin() + k, scored.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    float max_val = -INFINITY;
    for (int i = 0; i < k; i++)
        if (scored[i].first > max_val) max_val = scored[i].first;
    float sum_exp = 0;
    for (int i = 0; i < k; i++) {
        sum_exp += std::exp(scored[i].first - max_val);
    }
    
    float r = rng_.uniform();
    float cum = 0;
    for (int i = 0; i < k; i++) {
        cum += std::exp(scored[i].first - max_val) / sum_exp;
        if (r < cum) return scored[i].second;
    }
    return scored[k - 1].second;
}

int Sampler::sample_top_p(const float* logits, int vocab_size, float p, float temp) {
    if (temp < 1e-5f) return greedy(logits, vocab_size);
    
    std::vector<std::pair<float, int>> scored(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        scored[i] = {logits[i] / temp, i};
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    float max_val = scored[0].first;
    std::vector<float> probs(vocab_size);
    float sum_exp = 0;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(scored[i].first - max_val);
        sum_exp += probs[i];
    }
    
    float cum = 0;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum_exp;
        cum += probs[i];
        if (cum > p) { cutoff = i + 1; break; }
    }
    
    // Re-normalize over cutoff
    float sub_sum = 0;
    for (int i = 0; i < cutoff; i++) sub_sum += probs[i];
    
    float r = rng_.uniform();
    cum = 0;
    for (int i = 0; i < cutoff; i++) {
        cum += probs[i] / sub_sum;
        if (r < cum) return scored[i].second;
    }
    return scored[cutoff - 1].second;
}

int Sampler::sample(const float* logits, int vocab_size, const SamplerConfig& cfg,
                    const std::vector<int>& prev_tokens) {
    std::vector<float> adjusted(logits, logits + vocab_size);
    
    if (cfg.repetition_penalty > 1.0f) {
        apply_repetition_penalty(adjusted.data(), vocab_size, prev_tokens, cfg.repetition_penalty);
    }
    
    if (cfg.top_p < 1.0f) {
        return sample_top_p(adjusted.data(), vocab_size, cfg.top_p, cfg.temperature);
    }
    return sample_top_k(adjusted.data(), vocab_size, cfg.top_k, cfg.temperature);
}

} // namespace quant
