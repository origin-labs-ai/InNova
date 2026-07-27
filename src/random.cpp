#include "oil/random.h"
#include <cmath>
#include <limits>
#include <random>
#include <cstdlib>

namespace oil {

// ── GlobalSeedManager ──
uint64_t GlobalSeedManager::base_seed_ = 0;
uint64_t GlobalSeedManager::counter_ = 0;

uint64_t GlobalSeedManager::get_seed() {
    if (base_seed_ != 0) return base_seed_;
    const char* env = std::getenv("OIL_SEED");
    if (env) {
        base_seed_ = (uint64_t)std::atoll(env);
        if (base_seed_ == 0) base_seed_ = 1; // 0 means "use entropy"
    }
    if (base_seed_ == 0) {
        std::random_device rd;
        base_seed_ = ((uint64_t)rd() << 32) | rd();
        if (base_seed_ == 0) base_seed_ = 1;
    }
    return base_seed_;
}

uint64_t GlobalSeedManager::next_seed() {
    if (is_deterministic()) return base_seed_ + (++counter_);
    return get_seed() + (++counter_);
}

bool GlobalSeedManager::is_deterministic() {
    const char* env = std::getenv("OIL_SEED");
    return env != nullptr && std::atoll(env) != 0;
}

static inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

RNG::RNG(uint64_t seed_value) { seed(seed_value); }

void RNG::seed(uint64_t seed_value) {
    uint64_t tmp = seed_value;
    s[0] = splitmix64(tmp);
    s[1] = splitmix64(tmp);
}

uint64_t RNG::next_u64() {
    uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);
    return result;
}

uint32_t RNG::next_u32() { return (uint32_t)next_u64(); }

float RNG::uniform() {
    return (next_u64() >> 11) * (1.0f / 9007199254740992.0f);
}

float RNG::normal() {
    float u1 = uniform();
    float u2 = uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.283185307179586f * u2);
}

int RNG::uniform_int(int lo, int hi) {
    uint64_t range = (uint64_t)(hi - lo + 1);
    uint64_t limit = UINT64_MAX - (UINT64_MAX % range);
    uint64_t val;
    do { val = next_u64(); } while (val >= limit);
    return lo + (int)(val % range);
}

} // namespace oil
