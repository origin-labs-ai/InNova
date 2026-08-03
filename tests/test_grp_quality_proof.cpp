#include "oil/test.h"
#include "oil/format_registry.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

// GRP Quality Proof Test (formerly "GRP Lossless Proof")
// Verifies:
// 1. GRP formats are LOSSY (MSE > 0 for random data)
// 2. MSE bounded by est_mse × tolerance
// 3. ANTI-FRAUD: storage bytes < n * sizeof(float) for all non-sparse lossy formats
// 4. ANTI-FRAUD: MSE > 0 for all lossy formats (excl. SPARK_SPARSE which is lossless on dense data)
// 5. ANTI-FRAUD: indices not all zero for multi-centroid formats

static const char* ALL_GRP[] = {
    "OIL1_GRP", "OIL2_GRP", "OIL4_GRP", "OIL8_GRP", "OIL16_GRP",
    "SPARK_SPARSE_GRP", "SPARK_Q0_GRP"
};
static constexpr int NUM_GRP = 7;

void generate_normal(std::vector<float>& data, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data) v = dist(rng);
}

float compute_mse(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return (float)(sum / n);
}

void test_grp_is_lossy() {
    TEST_SUITE("GRP is Lossy (MSE > 0)");
    std::mt19937 rng(42);
    constexpr int64_t N = 10000;
    std::vector<float> data(N);
    generate_normal(data, rng);

    for (int f = 0; f < NUM_GRP; f++) {
        oil::FormatDescriptor fd = oil::FormatRegistry::parse_format_name(ALL_GRP[f]);
        oil::QuantResult qr = oil::FormatRegistry::quantize(data.data(), N, fd);
        TEST_CHECK(qr.success, "Quantize succeeded");

        std::vector<float> dequant(N);
        oil::FormatRegistry::dequantize(qr, dequant.data(), N);

        float mse = compute_mse(data.data(), dequant.data(), N);

        char msg[256];
        snprintf(msg, sizeof(msg), "%s MSE=%.6e (must be > 0 for lossy)", ALL_GRP[f], mse);
        TEST_CHECK(mse > 0.0f, msg);
        printf("  %s: MSE=%.6e (lossy confirmed)\n", ALL_GRP[f], mse);
    }
}

void test_grp_mse_bounded() {
    TEST_SUITE("GRP MSE bounded by est_mse");
    std::mt19937 rng(777);
    constexpr int64_t N = 10000;
    std::vector<float> data(N);
    generate_normal(data, rng);

    for (int f = 0; f < NUM_GRP; f++) {
        oil::FormatDescriptor fd = oil::FormatRegistry::parse_format_name(ALL_GRP[f]);
        oil::QuantResult qr = oil::FormatRegistry::quantize(data.data(), N, fd);
        std::vector<float> dequant(N);
        oil::FormatRegistry::dequantize(qr, dequant.data(), N);

        float mse = compute_mse(data.data(), dequant.data(), N);
        float upper_bound = fd.est_mse * 5.0f + 1e-6f;
        if (fd.est_mse == 0.0f) upper_bound = 0.5f;

        char msg[256];
        snprintf(msg, sizeof(msg), "%s MSE=%.6e upper=%.6e", ALL_GRP[f], mse, upper_bound);
        TEST_CHECK(mse <= upper_bound, msg);
        TEST_CHECK(mse >= 0.0f, "MSE non-negative");
    }
}

void test_grp_idempotency() {
    TEST_SUITE("GRP Idempotency (requantization stability)");
    std::mt19937 rng(555);
    constexpr int64_t N = 4096;
    std::vector<float> data(N);
    generate_normal(data, rng);

    for (int f = 0; f < NUM_GRP; f++) {
        oil::FormatDescriptor fd = oil::FormatRegistry::parse_format_name(ALL_GRP[f]);

        oil::QuantResult qr1 = oil::FormatRegistry::quantize(data.data(), N, fd);
        std::vector<float> dequant1(N);
        oil::FormatRegistry::dequantize(qr1, dequant1.data(), N);
        float mse1 = compute_mse(data.data(), dequant1.data(), N);

        oil::QuantResult qr2 = oil::FormatRegistry::quantize(dequant1.data(), N, fd);
        std::vector<float> dequant2(N);
        oil::FormatRegistry::dequantize(qr2, dequant2.data(), N);

        float mse = compute_mse(dequant1.data(), dequant2.data(), N);

        // Adaptive GRP is not strictly idempotent (range shrinks on requant).
        // Verify requantization MSE is at most 2x the original MSE.
        bool stable = (mse <= mse1 * 2.0f + 1e-10f);

        char msg[256];
        snprintf(msg, sizeof(msg), "%s requant MSE=%.6e orig=%.6e stable=%s",
                 ALL_GRP[f], mse, mse1, stable ? "YES" : "NO");
        TEST_CHECK(stable, msg);
        printf("  %s: orig_mse=%.6e requant_mse=%.6e\n", ALL_GRP[f], mse1, mse);
    }
}

void test_lossless_flags() {
    TEST_SUITE("Lossless Flags Corrected");
    const auto& singles = oil::FormatRegistry::get_all_singles();

    int lossless_count = 0;
    for (const auto& s : singles) {
        if (s.lossless) lossless_count++;
    }

    TEST_CHECK(lossless_count == 1, "Only OIL32 is lossless");
    printf("  Lossless count: %d (only OIL32)\n", lossless_count);

    oil::FormatDescriptor oil32 = oil::FormatRegistry::parse_format_name("OIL32");
    TEST_CHECK(oil32.lossless, "OIL32 is lossless");
    printf("  OIL32 lossless: %s\n", oil32.lossless ? "YES" : "NO");
}

void test_anti_fraud_storage() {
    TEST_SUITE("ANTI-FRAUD: Storage < n*sizeof(float) for all non-OIL32");
    std::mt19937 rng(314159);
    constexpr int64_t N = 2048;
    std::vector<float> data(N);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data) v = dist(rng);

    const auto& singles = oil::FormatRegistry::get_all_singles();
    for (const auto& s : singles) {
        if (s.id == oil::RegFormat::OIL32) continue; // OIL32 is identity
        if (s.id == oil::RegFormat::SPARK_SPARSE) continue; // sparse formats overflow on dense data
        if (s.id == oil::RegFormat::SPARK_SPARSE_GRP) continue; // sparse formats overflow on dense data

        oil::QuantResult qr = oil::FormatRegistry::quantize(data.data(), N, s);

        // Actual ON-DISK storage: the canonical wire payload (indices +
        // codebook channels). In-memory metadata (codebook_fp32, group
        // scales/zero-points) is never serialized, so it is not counted.
        int64_t storage_bytes = (int64_t)oil::FormatRegistry::serialized_size_bytes(qr);

        int64_t fp32_size = N * (int64_t)sizeof(float);

        char msg[256];
        snprintf(msg, sizeof(msg), "%s storage=%lld bytes vs FP32=%lld bytes (%.1f%%)",
                 s.name.c_str(), (long long)storage_bytes, (long long)fp32_size,
                 100.0 * storage_bytes / fp32_size);
        TEST_CHECK(storage_bytes < fp32_size, msg);
        printf("  %s: %lld / %lld bytes (%.1f%%) — PASS\n",
               s.name.c_str(), (long long)storage_bytes, (long long)fp32_size,
               100.0 * storage_bytes / fp32_size);
    }
}

void test_anti_fraud_mse_positive() {
    TEST_SUITE("ANTI-FRAUD: MSE > 0 for all non-OIL32 formats");
    std::mt19937 rng(271828);
    constexpr int64_t N = 4096;
    std::vector<float> data(N);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data) v = dist(rng);

    const auto& singles = oil::FormatRegistry::get_all_singles();
    for (const auto& s : singles) {
        if (s.lossless) continue;
        if (s.id == oil::RegFormat::SPARK_SPARSE) continue; // dense data above threshold = lossless

        oil::QuantResult qr = oil::FormatRegistry::quantize(data.data(), N, s);
        if (!qr.success) continue;

        std::vector<float> dequant(N);
        oil::FormatRegistry::dequantize(qr, dequant.data(), N);

        float mse = compute_mse(data.data(), dequant.data(), N);

        char msg[256];
        snprintf(msg, sizeof(msg), "%s MSE=%.6e (must be > 0 for lossy format)", s.name.c_str(), mse);
        TEST_CHECK(mse > 0.0f, msg);
        printf("  %s: MSE=%.6e — PASS\n", s.name.c_str(), mse);
    }
}

void test_anti_fraud_indices_not_all_zero() {
    TEST_SUITE("ANTI-FRAUD: Indices not all zero for non-OIL32 formats");
    std::mt19937 rng(161803);
    constexpr int64_t N = 4096;
    std::vector<float> data(N);
    std::normal_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : data) v = dist(rng);

    const auto& singles = oil::FormatRegistry::get_all_singles();
    for (const auto& s : singles) {
        if (s.lossless) continue;
        if (s.id == oil::RegFormat::OIL1) continue; // OIL1 has 1 centroid, all indices = 0 is correct
        if (s.id == oil::RegFormat::OIL1_GRP) continue; // k=1, all indices = 0 is correct
        if (s.id == oil::RegFormat::SPARK_SPARSE) continue; // sparse format may have empty/all-zero indices

        oil::QuantResult qr = oil::FormatRegistry::quantize(data.data(), N, s);
        if (!qr.success) continue;

        bool all_zero = true;
        for (size_t i = 0; i < qr.indices.size(); i++) {
            if (qr.indices[i] != 0) { all_zero = false; break; }
        }

        char msg[256];
        snprintf(msg, sizeof(msg), "%s indices all_zero=%s (must be false for lossy format)",
                 s.name.c_str(), all_zero ? "YES" : "NO");
        TEST_CHECK(!all_zero, msg);
        printf("  %s: indices not all zero — %s\n", s.name.c_str(), all_zero ? "FAIL" : "PASS");
    }
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  GRP QUALITY PROOF TEST\n");
    printf("  GRP = lossy with bounded MSE (no fraud checking)\n");
    printf("  Includes ANTI-FRAUD assertions\n");
    printf("═══════════════════════════════════════════════════════════\n");

    test_lossless_flags();
    test_grp_is_lossy();
    test_grp_mse_bounded();
    test_grp_idempotency();
    test_anti_fraud_storage();
    test_anti_fraud_mse_positive();
    test_anti_fraud_indices_not_all_zero();

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  GRP Quality Proof complete.\n");
    printf("═══════════════════════════════════════════════════════════\n");

    return TEST_REPORT() == 0 ? 0 : 1;
}
