#include "oil/test.h"
#include "oil/format_registry.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int64_t kPayloadElements = 16384;
constexpr const char* kFormats[] = {
    "OIL1", "OIL2", "OIL4", "OIL8", "OIL16", "OIL32",
    "OIL1_GRP", "OIL2_GRP", "OIL4_GRP", "OIL8_GRP", "OIL16_GRP",
    "SPARK_SPARSE", "SPARK_SPARSE_GRP", "SPARK_Q0", "SPARK_Q0_GRP",
};

void make_grouped_weights(std::vector<float>& values) {
    std::mt19937 rng(20260801);
    for (int64_t group = 0; group < kPayloadElements / 1024; ++group) {
        const float scale = 0.20f + 0.15f * static_cast<float>(group + 1);
        std::normal_distribution<float> distribution(0.0f, scale);
        for (int64_t local = 0; local < 1024; ++local) {
            values[static_cast<size_t>(group * 1024 + local)] = distribution(rng);
        }
    }
}

size_t claimed_bytes(const oil::FormatDescriptor& format) {
    return static_cast<size_t>(std::ceil(format.bpw * static_cast<float>(kPayloadElements) / 8.0f));
}

void test_claimed_bpw_is_wire_bpw() {
    TEST_SUITE("Native BPW Contract: claimed equals stored payload");
    std::vector<float> input(static_cast<size_t>(kPayloadElements));
    make_grouped_weights(input);

    for (const char* name : kFormats) {
        const oil::FormatDescriptor format = oil::FormatRegistry::parse_format_name(name);
        const oil::QuantResult result = oil::FormatRegistry::quantize(input.data(), kPayloadElements, format);
        TEST_CHECK(result.success, "Quantize succeeded");

        const size_t stored = oil::FormatRegistry::serialized_size_bytes(result);
        const size_t cap = claimed_bytes(format);
        const float actual = oil::FormatRegistry::actual_bpw(result);
        char message[256];
        std::snprintf(message, sizeof(message), "%s stores %zu bytes (cap %zu, actual %.3f BPW, claimed %.3f BPW)",
                      name, stored, cap, actual, format.bpw);
        TEST_CHECK(stored <= cap, message);
        TEST_CHECK(actual <= format.bpw + 1e-5f, message);

        std::vector<float> decoded(static_cast<size_t>(kPayloadElements));
        oil::FormatRegistry::dequantize(result, decoded.data(), kPayloadElements);
        bool finite = true;
        for (float value : decoded) finite = finite && std::isfinite(value);
        TEST_CHECK(finite, "Decoded values are finite");
        if (format.lossless) {
            TEST_CHECK(std::memcmp(input.data(), decoded.data(), input.size() * sizeof(float)) == 0,
                       "OIL32 remains bitwise lossless");
        }
        std::printf("  %-18s %6zu bytes  %5.2f / %5.2f BPW\n", name, stored, actual, format.bpw);
    }
}

float descriptor_bpw(oil::RegFormat format) {
    return oil::FormatRegistry::parse_format_name(oil::format_name(oil::regformat_to_format(format))).bpw;
}

void test_mix_descriptors_are_true_weighted_sums() {
    TEST_SUITE("Native BPW Contract: mix descriptors are weighted sums");
    const auto check_mix = [](const oil::MixDescriptor& mix) {
        const float expected = mix.tier1_ratio * descriptor_bpw(mix.tier1_fmt) +
                               mix.tier2_ratio * descriptor_bpw(mix.tier2_fmt) +
                               mix.tier3_ratio * descriptor_bpw(mix.tier3_fmt) +
                               mix.tier4_ratio * descriptor_bpw(mix.tier4_fmt);
        char message[256];
        std::snprintf(message, sizeof(message), "%s %.3f BPW must equal weighted %.3f BPW",
                      mix.name.c_str(), mix.effective_bpw, expected);
        TEST_CHECK_CLOSE(mix.effective_bpw, expected, 0.0001f, message);
    };
    for (const auto& mix : oil::FormatRegistry::get_all_twi_mixes()) check_mix(mix);
    for (const auto& mix : oil::FormatRegistry::get_all_four_mixes()) check_mix(mix);
}

void test_quad_claimed_bpw() {
    TEST_SUITE("Native BPW Contract: QUAD mixes match documented claims");
    const auto find_quad = [](const char* name) -> const oil::MixDescriptor* {
        for (const auto& mix : oil::FormatRegistry::get_all_four_mixes())
            if (mix.name == name) return &mix;
        return nullptr;
    };
    const auto check_quad = [&find_quad](const char* name, float claimed) {
        const oil::MixDescriptor* mix = find_quad(name);
        TEST_CHECK(mix != nullptr, "QUAD mix exists");
        if (!mix) return;
        char message[256];
        std::snprintf(message, sizeof(message), "%s effective BPW %.3f equals claimed %.3f",
                      name, mix->effective_bpw, claimed);
        TEST_CHECK_CLOSE(mix->effective_bpw, claimed, 0.001f, message);
        const float ratio_sum = mix->tier1_ratio + mix->tier2_ratio +
                                mix->tier3_ratio + mix->tier4_ratio;
        std::snprintf(message, sizeof(message), "%s tier ratios sum to 1.0 (%.4f)",
                      name, ratio_sum);
        TEST_CHECK_CLOSE(ratio_sum, 1.0f, 0.0001f, message);
        for (float ratio : { mix->tier1_ratio, mix->tier2_ratio, mix->tier3_ratio, mix->tier4_ratio })
            TEST_CHECK(ratio >= 0.0f && ratio <= 1.0f, "tier ratios in [0,1]");
    };
    check_quad("QUAD_OIL2_OIL4_OIL8_OIL16", 2.92f);
    check_quad("QUAD_OIL4_OIL8_OIL16_OIL32", 5.84f);
}

void test_spark_claims_match_layout() {
    TEST_SUITE("Native BPW Contract: SPARK layouts");
    const auto q0 = oil::FormatRegistry::parse_format_name("SPARK_Q0");
    const auto q0_grouped = oil::FormatRegistry::parse_format_name("SPARK_Q0_GRP");
    const auto sparse = oil::FormatRegistry::parse_format_name("SPARK_SPARSE");
    TEST_CHECK_CLOSE(q0.bpw, 1.5f, 0.0001f, "SPARK_Q0 is 1.50 BPW");
    TEST_CHECK_CLOSE(q0_grouped.bpw, 1.5f, 0.0001f, "SPARK_Q0_GRP is 1.50 BPW");
    TEST_CHECK_CLOSE(sparse.bpw, 2.0f, 0.0001f, "SPARK_SPARSE is 2.00 BPW");
    TEST_CHECK_CLOSE(oil::format_bpw(oil::Format::SPARK_Q0), q0.bpw, 0.0001f,
                     "Public SPARK_Q0 BPW agrees with registry");
    TEST_CHECK_CLOSE(oil::format_bpw(oil::Format::SPARK_Q0_GRP), q0_grouped.bpw, 0.0001f,
                     "Public SPARK_Q0_GRP BPW agrees with registry");
}

void test_selection_never_exceeds_target() {
    TEST_SUITE("Native BPW Contract: selection never exceeds target");
    std::vector<float> values(2048, 0.25f);
    const auto selected = oil::FormatRegistry::select_best_format(2.0f, values.data(), static_cast<int64_t>(values.size()));
    TEST_CHECK(selected.bpw <= 2.0f + 1e-6f, "Selected single format stays within target");
    const auto forced = oil::FormatRegistry::apply_forced_distribution(4.0f, 4, values.data(), static_cast<int64_t>(values.size()));
    for (const auto& format : forced) {
        TEST_CHECK(format.bpw <= 4.0f + 1e-6f, "Forced format stays within target");
    }
}

} // namespace

int main() {
    std::printf("NATIVE BPW PROOF: payload bytes, quality-safe decoding, and mix arithmetic\n");
    test_claimed_bpw_is_wire_bpw();
    test_mix_descriptors_are_true_weighted_sums();
    test_quad_claimed_bpw();
    test_spark_claims_match_layout();
    test_selection_never_exceeds_target();
    return TEST_REPORT() == 0 ? 0 : 1;
}
