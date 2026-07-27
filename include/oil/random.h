#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace oil {

// ── GlobalSeedManager — entropy-based seed source ──
// Uses std::random_device by default. Override with OIL_SEED env var.
// OIL_SEED=0 disables entropy for full deterministic reproducibility.
class GlobalSeedManager {
public:
    static uint64_t get_seed();
    static uint64_t next_seed();
    static bool is_deterministic();

private:
    static uint64_t base_seed_;
    static uint64_t counter_;
};

class RNG {
public:
    explicit RNG(uint64_t seed = GlobalSeedManager::get_seed());

    void seed(uint64_t s);

    float uniform();
    float normal();
    int uniform_int(int lo, int hi);

    uint64_t next_u64();
    uint32_t next_u32();

private:
    uint64_t s[2];

    static uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

} // namespace oil
