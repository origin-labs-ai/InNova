#include "adapters/delta_adapter.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace oil {
namespace adapters {

DeltaAdapterHost::DeltaAdapterHost(const DeltaAdapterConfig& cfg)
    : cfg_(cfg) {}

std::uint32_t DeltaAdapterHost::create_adapter(const std::string& task_name) {
    std::uint32_t id = next_id_++;
    DeltaAdapter adapter;
    adapter.task_id = id;
    adapter.task_name = task_name;
    adapter.frozen = false;
    adapter.lambda = cfg_.default_lambda;
    adapter.total_params = 0;
    adapters_[id] = std::move(adapter);
    return id;
}

void DeltaAdapterHost::freeze_adapter(std::uint32_t adapter_id) {
    auto it = adapters_.find(adapter_id);
    if (it != adapters_.end()) {
        it->second.frozen = true;
    }
}

void DeltaAdapterHost::unfreeze_adapter(std::uint32_t adapter_id) {
    auto it = adapters_.find(adapter_id);
    if (it != adapters_.end()) {
        it->second.frozen = false;
    }
}

bool DeltaAdapterHost::remove_adapter(std::uint32_t adapter_id) {
    return adapters_.erase(adapter_id) > 0;
}

void DeltaAdapterHost::update_delta(std::uint32_t adapter_id,
                                     const std::string& layer_name,
                                     const float* gradient, std::size_t n,
                                     float lr) {
    auto it = adapters_.find(adapter_id);
    if (it == adapters_.end() || it->second.frozen) return;

    auto& delta = it->second.layers[layer_name];
    if (delta.delta.size() != n) {
        delta.delta.assign(n, 0.0f);
    }
    for (std::size_t i = 0; i < n; ++i) {
        delta.delta[i] -= lr * gradient[i];
    }
    it->second.total_params = 0;
    for (const auto& [name, wd] : it->second.layers) {
        it->second.total_params += wd.delta.size();
    }
}

void DeltaAdapterHost::accumulate_fisher(std::uint32_t adapter_id,
                                          const std::string& layer_name,
                                          const float* grad_sq, std::size_t n) {
    auto it = adapters_.find(adapter_id);
    if (it == adapters_.end()) return;

    auto& delta = it->second.layers[layer_name];
    if (delta.fisher_diag.size() != n) {
        delta.fisher_diag.assign(n, 0.0f);
    }
    const float mom = 0.9f;
    for (std::size_t i = 0; i < n; ++i) {
        delta.fisher_diag[i] = mom * delta.fisher_diag[i] + (1.0f - mom) * grad_sq[i];
    }
}

void DeltaAdapterHost::merge_deltas(float* base_weights,
                                     const std::string& layer_name,
                                     std::size_t n,
                                     std::uint32_t adapter_id) const {
    auto it = adapters_.find(adapter_id);
    if (it == adapters_.end() || it->second.frozen) return;

    auto lit = it->second.layers.find(layer_name);
    if (lit == it->second.layers.end()) return;

    const auto& delta = lit->second.delta;
    std::size_t sz = (std::min)(n, delta.size());
    for (std::size_t i = 0; i < sz; ++i) {
        base_weights[i] += delta[i];
    }
}

void DeltaAdapterHost::merge_all_active(float* base_weights,
                                         const std::string& layer_name,
                                         std::size_t n) const {
    for (const auto& [id, adapter] : adapters_) {
        if (adapter.frozen) continue;
        auto lit = adapter.layers.find(layer_name);
        if (lit == adapter.layers.end()) continue;

        const auto& delta = lit->second.delta;
        std::size_t sz = (std::min)(n, delta.size());
        for (std::size_t i = 0; i < sz; ++i) {
            base_weights[i] += adapter.lambda * delta[i];
        }
    }
}

float DeltaAdapterHost::regularizer(std::uint32_t adapter_id, float lambda) const {
    auto it = adapters_.find(adapter_id);
    if (it == adapters_.end()) return 0.0f;

    double acc = 0.0;
    for (const auto& [name, wd] : it->second.layers) {
        for (std::size_t i = 0; i < wd.delta.size(); ++i) {
            double f = (i < wd.fisher_diag.size()) ? wd.fisher_diag[i] : 0.0;
            acc += f * static_cast<double>(wd.delta[i]) * static_cast<double>(wd.delta[i]);
        }
    }
    return static_cast<float>(lambda * acc);
}

bool DeltaAdapterHost::is_frozen(std::uint32_t adapter_id) const {
    auto it = adapters_.find(adapter_id);
    return (it != adapters_.end()) ? it->second.frozen : true;
}

std::size_t DeltaAdapterHost::total_adapter_params() const {
    std::size_t total = 0;
    for (const auto& [id, adapter] : adapters_) {
        total += adapter.total_params;
    }
    return total;
}

DeltaAdapter* DeltaAdapterHost::get_adapter(std::uint32_t adapter_id) {
    auto it = adapters_.find(adapter_id);
    return (it != adapters_.end()) ? &it->second : nullptr;
}

const DeltaAdapter* DeltaAdapterHost::get_adapter(std::uint32_t adapter_id) const {
    auto it = adapters_.find(adapter_id);
    return (it != adapters_.end()) ? &it->second : nullptr;
}

std::vector<std::uint32_t> DeltaAdapterHost::active_adapter_ids() const {
    std::vector<std::uint32_t> ids;
    for (const auto& [id, adapter] : adapters_) {
        if (!adapter.frozen) ids.push_back(id);
    }
    return ids;
}

} // namespace adapters
} // namespace oil
