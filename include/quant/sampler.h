#pragma once

#include "quant/random.h"
#include <cstdint>
#include <vector>

namespace quant {

struct SamplerConfig {
    float temperature = 1.0f;
    int top_k = 40;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    int max_tokens = 2048;
};

class Sampler {
public:
    explicit Sampler(uint64_t seed = 42);

    int greedy(const float* logits, int vocab_size);
    int sample_top_k(const float* logits, int vocab_size, int k, float temp);
    int sample_top_p(const float* logits, int vocab_size, float p, float temp);
    int sample(const float* logits, int vocab_size, const SamplerConfig& cfg,
               const std::vector<int>& prev_tokens = {});

    void apply_temperature(float* logits, int n, float temp) const;
    void apply_repetition_penalty(float* logits, int n,
                                   const std::vector<int>& prev_tokens,
                                   float penalty) const;

private:
    RNG rng_;
};

} // namespace quant
