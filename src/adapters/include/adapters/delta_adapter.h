#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include "oil/types.h"
#include "oil/format_registry.h"

namespace oil {
namespace adapters {

struct DeltaAdapter {
    std::uint32_t task_id = 0;
    std::string task_name;
    bool frozen = false;

    struct WeightDelta {
        std::vector<float> delta;
        std::vector<float> fisher_diag;
        float importance = 0.0f;
    };

    std::unordered_map<std::string, WeightDelta> layers;
    std::size_t total_params = 0;
    float lambda = 1.0f;
};

struct DeltaAdapterConfig {
    std::size_t max_adapters = 16;
    float default_lambda = 1.0f;
    float freeze_fisher_threshold = 0.5f;
    bool quantize_deltas = true;
    float delta_target_bpw = 2.0f;
};

class DeltaAdapterHost {
public:
    explicit DeltaAdapterHost(const DeltaAdapterConfig& cfg = {});
    ~DeltaAdapterHost() = default;

    std::uint32_t create_adapter(const std::string& task_name);
    void freeze_adapter(std::uint32_t adapter_id);
    void unfreeze_adapter(std::uint32_t adapter_id);
    bool remove_adapter(std::uint32_t adapter_id);

    void update_delta(std::uint32_t adapter_id, const std::string& layer_name,
                      const float* gradient, std::size_t n, float lr);
    void accumulate_fisher(std::uint32_t adapter_id, const std::string& layer_name,
                           const float* grad_sq, std::size_t n);

    void merge_deltas(float* base_weights, const std::string& layer_name,
                      std::size_t n, std::uint32_t adapter_id) const;
    void merge_all_active(float* base_weights, const std::string& layer_name,
                          std::size_t n) const;

    [[nodiscard]] float regularizer(std::uint32_t adapter_id, float lambda) const;
    [[nodiscard]] bool is_frozen(std::uint32_t adapter_id) const;
    [[nodiscard]] std::size_t adapter_count() const { return adapters_.size(); }
    [[nodiscard]] std::size_t total_adapter_params() const;

    DeltaAdapter* get_adapter(std::uint32_t adapter_id);
    const DeltaAdapter* get_adapter(std::uint32_t adapter_id) const;
    std::vector<std::uint32_t> active_adapter_ids() const;

private:
    DeltaAdapterConfig cfg_;
    std::unordered_map<std::uint32_t, DeltaAdapter> adapters_;
    std::uint32_t next_id_ = 1;
};

} // namespace adapters
} // namespace oil
