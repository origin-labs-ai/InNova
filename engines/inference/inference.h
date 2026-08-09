#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "quant/model.h"
#include "quant/kv_cache.h"
#include "quant/sampler.h"
#include "quant/tokenizer.h"
#include "quant/generator.h"

namespace quant {
namespace engine {

struct EngineConfig {
    int64_t max_seq_len = 2048;
    int num_threads = 1;
    bool use_fp16 = false;
    bool use_kv_cache = true;
    SamplerConfig sampler;
};

struct InferenceStats {
    int tokens_generated;
    float duration_sec;
    float tokens_per_sec;
    size_t peak_memory_bytes;
    float kv_cache_utilization;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Load model from .quant file
    Status load(const std::string& quant_path);

    // Initialize with model and tokenizer
    void init(Model* model, Tokenizer* tokenizer, const EngineConfig& cfg = EngineConfig());

    // Single prompt → response
    std::string generate(const std::string& prompt);

    // With stats
    std::pair<std::string, InferenceStats> generate_with_stats(const std::string& prompt);

    // Streaming generation
    void generate_stream(const std::string& prompt,
                         std::function<void(const std::string&)> on_token);

    // Batch generation
    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts);

    // Reset context (clear KV cache)
    void reset_context();

    // Getters
    const EngineConfig& config() const;
    InferenceStats last_stats() const;

private:
    Model* model_ = nullptr;
    Tokenizer* tokenizer_ = nullptr;
    EngineConfig config_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<Generator> generator_;
    InferenceStats stats_{};
    bool owns_model_ = false;
    bool owns_tokenizer_ = false;

    void update_stats(int tokens, float duration);
};

} // namespace engine
} // namespace quant
