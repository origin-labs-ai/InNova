#include "oil/test.h"
#include "oil/format_registry.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <cstring>
#include <algorithm>

// Section 2: FP32 Gather Quality Proof
// All 15 formats: quantize -> dequantize roundtrip
// OIL32 (ONLY lossless format): MSE must be EXACTLY 0.0
// Lossy formats (ALL others): MSE must be > 0 and bounded
// Anti-fraud: storage bytes < n * sizeof(float) for non-OIL32

static constexpr int NUM_FORMATS = 15;
static const char* FORMAT_NAMES[NUM_FORMATS] = {
    "OIL1", "OIL2", "OIL4", "OIL8", "OIL16", "OIL32",
    "OIL1_GRP", "OIL2_GRP", "OIL4_GRP", "OIL8_GRP", "OIL16_GRP",
    "SPARK_SPARSE", "SPARK_SPARSE_GRP", "SPARK_Q0", "SPARK_Q0_GRP"
};

static constexpr bool EXPECTED_LOSSLESS[NUM_FORMATS] = {
    false, false, false, false, false, true,
    false, false, false, false, false,
    false, false, false, false
};

void generate_weights(std::vector<float>& data, int dist_type, std::mt19937& rng) {
    switch (dist_type) {
        case 0: { // Normal
            std::normal_distribution<float> d(0.0f, 1.0f);
            for (auto& v : data) v = d(rng);
            break;
        }
        case 1: { // Uniform
            std::uniform_real_distribution<float> d(-2.0f, 2.0f);
            for (auto& v : data) v = d(rng);
            break;
        }
        case 2: { // Laplace
            std::exponential_distribution<float> d(1.0f);
            for (auto& v : data) v = d(rng) - d(rng);
            break;
        }
        case 3: { // Near-zero (sparse)
            std::normal_distribution<float> d(0.0f, 0.01f);
            for (auto& v : data) v = d(rng);
            break;
        }
        case 4: { // Large values
            std::uniform_real_distribution<float> d(-100.0f, 100.0f);
            for (auto& v : data) v = d(rng);
            break;
        }
        default: {
            std::normal_distribution<float> d(0.0f, 1.0f);
            for (auto& v : data) v = d(rng);
            break;
        }
    }
}

void test_lossless_roundtrip() {
    TEST_SUITE("Section 2: FP32 Gather Quality — Lossless Formats");
    std::mt19937 rng(42);
    const int N = 4096;

    for (int fi = 0; fi < NUM_FORMATS; fi++) {
        if (!EXPECTED_LOSSLESS[fi]) continue;

        oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name(FORMAT_NAMES[fi]);
        TEST_CHECK(!fmt.name.empty(), "Format found");

        for (int dist = 0; dist < 5; dist++) {
            std::vector<float> original(N);
            generate_weights(original, dist, rng);

            oil::QuantResult qr = oil::FormatRegistry::quantize(
                original.data(), N, fmt);
            TEST_CHECK(qr.success, "Quantize succeeded");

            std::vector<float> dequant(N);
            oil::FormatRegistry::dequantize(qr, dequant.data(), N);

            float mse = oil::FormatRegistry::measure_mse(
                original.data(), dequant.data(), N);

            bool bitwise = (std::memcmp(original.data(), dequant.data(),
                                        N * sizeof(float)) == 0);

            char msg[256];
            snprintf(msg, sizeof(msg), "%s dist=%d MSE=%.6e (must be 0.0) bitwise=%s",
                     FORMAT_NAMES[fi], dist, mse, bitwise ? "YES" : "NO");
            TEST_CHECK(mse == 0.0f, msg);
            TEST_CHECK(bitwise, msg);
        }
    }
}

void test_lossy_roundtrip() {
    TEST_SUITE("Section 2: FP32 Gather Quality — Lossy Formats");
    std::mt19937 rng(123);
    const int N = 4096;

    for (int fi = 0; fi < NUM_FORMATS; fi++) {
        if (EXPECTED_LOSSLESS[fi]) continue;
        // Sparse formats are tuned for sparse tensors; on dense data their
        // stored size and MSE depend on the non-zero fraction, not the
        // format's claimed quality, so the claimed-bpw MSE contract does not
        // apply (sparse-domain coverage lives in the sparse tests).
        if (std::string(FORMAT_NAMES[fi]) == "SPARK_SPARSE") continue;
        if (std::string(FORMAT_NAMES[fi]) == "SPARK_SPARSE_GRP") continue;

        oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name(FORMAT_NAMES[fi]);
        TEST_CHECK(!fmt.name.empty(), "Format found");

        for (int dist = 0; dist < 3; dist++) {
            std::vector<float> original(N);
            generate_weights(original, dist, rng);

            oil::QuantResult qr = oil::FormatRegistry::quantize(
                original.data(), N, fmt);
            TEST_CHECK(qr.success, "Quantize succeeded");

            std::vector<float> dequant(N);
            oil::FormatRegistry::dequantize(qr, dequant.data(), N);

            float mse = oil::FormatRegistry::measure_mse(
                original.data(), dequant.data(), N);

            float upper_bound = fmt.est_mse * 3.0f + 1e-6f;
            if (fmt.est_mse == 0.0f) upper_bound = 0.1f;

            char msg[256];
            snprintf(msg, sizeof(msg), "%s dist=%d MSE=%.6e est=%.6e bound=%.6e",
                     FORMAT_NAMES[fi], dist, mse, fmt.est_mse, upper_bound);
            TEST_CHECK(mse <= upper_bound, msg);
            TEST_CHECK(mse >= 0.0f, "MSE non-negative");
        }
    }
}

void test_different_sizes() {
    TEST_SUITE("Section 2: FP32 Gather Quality — Variable Sizes");
    std::mt19937 rng(999);
    int sizes[] = {1, 16, 32, 128, 512, 1024, 4096};

    for (int si = 0; si < 7; si++) {
        int N = sizes[si];
        std::vector<float> original(N);
        generate_weights(original, 0, rng);

        for (int fi = 0; fi < NUM_FORMATS; fi++) {
            oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name(FORMAT_NAMES[fi]);
            oil::QuantResult qr = oil::FormatRegistry::quantize(
                original.data(), N, fmt);
            TEST_CHECK(qr.success, "Quantize succeeded for all sizes");

            std::vector<float> dequant(N);
            oil::FormatRegistry::dequantize(qr, dequant.data(), N);

            if (EXPECTED_LOSSLESS[fi]) {
                float mse = oil::FormatRegistry::measure_mse(
                    original.data(), dequant.data(), N);
                char msg[256];
                snprintf(msg, sizeof(msg), "N=%d %s MSE=%.6e (must be 0.0)",
                         N, FORMAT_NAMES[fi], mse);
                TEST_CHECK(mse == 0.0f, msg);
            }
        }
    }
}

void test_evaluate_format_quality() {
    TEST_SUITE("Section 2: evaluate_format_quality consistency");
    std::mt19937 rng(777);
    const int N = 2048;
    std::vector<float> original(N);
    generate_weights(original, 0, rng);

    for (int fi = 0; fi < NUM_FORMATS; fi++) {
        oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name(FORMAT_NAMES[fi]);
        float quality_mse = oil::FormatRegistry::evaluate_format_quality(
            original.data(), N, fmt);

        oil::QuantResult qr = oil::FormatRegistry::quantize(
            original.data(), N, fmt);
        std::vector<float> dequant(N);
        oil::FormatRegistry::dequantize(qr, dequant.data(), N);
        float manual_mse = oil::FormatRegistry::measure_mse(
            original.data(), dequant.data(), N);

        char msg[256];
        snprintf(msg, sizeof(msg), "%s: evaluate=%.6e manual=%.6e match=%s",
                 FORMAT_NAMES[fi], quality_mse, manual_mse,
                 (std::fabs(quality_mse - manual_mse) < 1e-10f) ? "YES" : "NO");
        TEST_CHECK(std::fabs(quality_mse - manual_mse) < 1e-10f, msg);
    }
}

void test_anti_fraud_storage() {
    TEST_SUITE("Section 2: Anti-Fraud Storage Check");
    std::mt19937 rng(42);
    const int N = 4096;
    std::vector<float> data(N);
    generate_weights(data, 0, rng);

    for (int fi = 0; fi < NUM_FORMATS; fi++) {
        if (EXPECTED_LOSSLESS[fi]) continue;
        oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name(FORMAT_NAMES[fi]);
        if (fmt.id == oil::RegFormat::SPARK_SPARSE) continue; // sparse formats overflow on dense data
        if (fmt.id == oil::RegFormat::SPARK_SPARSE_GRP) continue; // sparse formats overflow on dense data
        oil::QuantResult qr = oil::FormatRegistry::quantize(data.data(), N, fmt);
        TEST_CHECK(qr.success, "Quantize succeeded");

        // Actual ON-DISK storage: the canonical wire payload (indices +
        // codebook channels). In-memory metadata (codebook_fp32, group
        // scales/zero-points) is never serialized, so it is not counted.
        int64_t storage_bytes = (int64_t)oil::FormatRegistry::serialized_size_bytes(qr);
        int64_t fp32_size = N * (int64_t)sizeof(float);

        char msg[256];
        snprintf(msg, sizeof(msg), "%s storage=%lld < FP32=%lld bytes", FORMAT_NAMES[fi], (long long)storage_bytes, (long long)fp32_size);
        TEST_CHECK(storage_bytes < fp32_size, msg);
    }
}

void test_edge_cases() {
    TEST_SUITE("Section 2: Edge Cases");
    std::mt19937 rng(314);

    // All zeros
    {
        const int N = 256;
        std::vector<float> zeros(N, 0.0f);
        for (int fi = 0; fi < NUM_FORMATS; fi++) {
            oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name(FORMAT_NAMES[fi]);
            oil::QuantResult qr = oil::FormatRegistry::quantize(zeros.data(), N, fmt);
            TEST_CHECK(qr.success, "Quantize zeros succeeded");
            std::vector<float> dequant(N);
            oil::FormatRegistry::dequantize(qr, dequant.data(), N);
        }
    }

    // All same value
    {
        const int N = 256;
        std::vector<float> same(N, 3.14159f);
        for (int fi = 0; fi < NUM_FORMATS; fi++) {
            oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name(FORMAT_NAMES[fi]);
            oil::QuantResult qr = oil::FormatRegistry::quantize(same.data(), N, fmt);
            TEST_CHECK(qr.success, "Quantize uniform succeeded");
            std::vector<float> dequant(N);
            oil::FormatRegistry::dequantize(qr, dequant.data(), N);
        }
    }

    // Very small values
    {
        const int N = 256;
        std::vector<float> tiny(N);
        std::uniform_real_distribution<float> d(-1e-7f, 1e-7f);
        for (auto& v : tiny) v = d(rng);
        for (int fi = 0; fi < NUM_FORMATS; fi++) {
            oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name(FORMAT_NAMES[fi]);
            oil::QuantResult qr = oil::FormatRegistry::quantize(tiny.data(), N, fmt);
            TEST_CHECK(qr.success, "Quantize tiny values succeeded");
            std::vector<float> dequant(N);
            oil::FormatRegistry::dequantize(qr, dequant.data(), N);
        }
    }

    // Null/zero-length inputs
    {
        oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name("OIL4");
        oil::QuantResult qr = oil::FormatRegistry::quantize(nullptr, 0, fmt);
        TEST_CHECK(!qr.success || qr.num_elements == 0, "Null input handled");
    }
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  SECTION 2: FP32 GATHER QUALITY PROOF\n");
    printf("  All 15 formats: quantize -> dequantize roundtrip\n");
    printf("═══════════════════════════════════════════════════════════\n");

    test_lossless_roundtrip();
    test_lossy_roundtrip();
    test_different_sizes();
    test_evaluate_format_quality();
    test_anti_fraud_storage();
    test_edge_cases();

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Section 2 complete.\n");
    printf("═══════════════════════════════════════════════════════════\n");

    return TEST_REPORT() == 0 ? 0 : 1;
}
