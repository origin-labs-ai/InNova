#include "oil/test.h"
#include "oil/format_registry.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>

// Section 3: BPW 2.00 Proof
// Verifies SPARK_Q0 and SPARK_SPARSE both claim 2.0 BPW
// Verifies compute_average_bpw correctness
// Verifies mixed-precision BPW calculations

void test_spark_q0_bpw() {
    TEST_SUITE("Section 3: SPARK_Q0 BPW");
    oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name("SPARK_Q0");
    TEST_CHECK(!fmt.name.empty(), "SPARK_Q0 found");
    TEST_CHECK_CLOSE(fmt.bpw, 2.0f, 0.01f, "SPARK_Q0 BPW == 2.0");

    printf("  SPARK_Q0 BPW: %.2f\n", fmt.bpw);
}

void test_spark_sparse_bpw() {
    TEST_SUITE("Section 3: SPARK_SPARSE BPW");
    oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name("SPARK_SPARSE");
    TEST_CHECK(!fmt.name.empty(), "SPARK_SPARSE found");
    TEST_CHECK_CLOSE(fmt.bpw, 2.0f, 0.01f, "SPARK_SPARSE BPW == 2.0");

    printf("  SPARK_SPARSE BPW: %.2f\n", fmt.bpw);
}

void test_spark_q0_grp_bpw() {
    TEST_SUITE("Section 3: SPARK_Q0_GRP BPW");
    oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name("SPARK_Q0_GRP");
    TEST_CHECK(!fmt.name.empty(), "SPARK_Q0_GRP found");
    TEST_CHECK_CLOSE(fmt.bpw, 2.0f, 0.01f, "SPARK_Q0_GRP BPW == 2.0");

    printf("  SPARK_Q0_GRP BPW: %.2f\n", fmt.bpw);
}

void test_spark_sparse_grp_bpw() {
    TEST_SUITE("Section 3: SPARK_SPARSE_GRP BPW");
    oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name("SPARK_SPARSE_GRP");
    TEST_CHECK(!fmt.name.empty(), "SPARK_SPARSE_GRP found");
    TEST_CHECK_CLOSE(fmt.bpw, 2.0f, 0.01f, "SPARK_SPARSE_GRP BPW == 2.0");

    printf("  SPARK_SPARSE_GRP BPW: %.2f\n", fmt.bpw);
}

void test_compute_average_bpw_single_format() {
    TEST_SUITE("Section 3: compute_average_bpw single format");
    oil::FormatDescriptor fmt = oil::FormatRegistry::parse_format_name("OIL8");

    std::vector<oil::FormatDescriptor> assignment = {fmt};
    float avg = oil::FormatRegistry::compute_average_bpw(assignment);
    TEST_CHECK_CLOSE(avg, 8.0f, 0.01f, "Single OIL8 avg_bpw == 8.0");

    printf("  OIL8 single avg_bpw: %.2f\n", avg);
}

void test_compute_average_bpw_mixed() {
    TEST_SUITE("Section 3: compute_average_bpw mixed precision");
    oil::FormatDescriptor oil8 = oil::FormatRegistry::parse_format_name("OIL8");
    oil::FormatDescriptor oil2 = oil::FormatRegistry::parse_format_name("OIL2");
    oil::FormatDescriptor oil4 = oil::FormatRegistry::parse_format_name("OIL4");

    // 50% OIL8, 25% OIL2, 25% OIL4
    std::vector<oil::FormatDescriptor> assignment;
    for (int i = 0; i < 4; i++) {
        if (i == 0) assignment.push_back(oil8);
        else if (i == 1) assignment.push_back(oil2);
        else assignment.push_back(oil4);
    }

    float avg = oil::FormatRegistry::compute_average_bpw(assignment);
    float expected = (8.0f + 2.0f + 4.0f + 4.0f) / 4.0f;
    TEST_CHECK_CLOSE(avg, expected, 0.01f, "Mixed avg_bpw correct");

    printf("  Mixed avg_bpw: %.2f (expected %.2f)\n", avg, expected);
}

void test_compute_average_bpw_empty() {
    TEST_SUITE("Section 3: compute_average_bpw empty");
    std::vector<oil::FormatDescriptor> empty;
    float avg = oil::FormatRegistry::compute_average_bpw(empty);
    TEST_CHECK_CLOSE(avg, 0.0f, 0.01f, "Empty avg_bpw == 0.0");
}

void test_compute_average_bpw_all_2_00() {
    TEST_SUITE("Section 3: compute_average_bpw all-2.00 assignment");
    oil::FormatDescriptor spark_q0 = oil::FormatRegistry::parse_format_name("SPARK_Q0");
    oil::FormatDescriptor spark_sparse = oil::FormatRegistry::parse_format_name("SPARK_SPARSE");

    std::vector<oil::FormatDescriptor> assignment = {spark_q0, spark_sparse};
    float avg = oil::FormatRegistry::compute_average_bpw(assignment);
    TEST_CHECK_CLOSE(avg, 2.0f, 0.01f, "All-2.00 avg_bpw == 2.0");

    printf("  All-2.00 avg_bpw: %.2f\n", avg);
}

void test_compute_average_bpw_weighted_distribution() {
    TEST_SUITE("Section 3: compute_average_bpw weighted distribution");
    oil::FormatDescriptor oil1 = oil::FormatRegistry::parse_format_name("OIL1");
    oil::FormatDescriptor oil32 = oil::FormatRegistry::parse_format_name("OIL32");

    // 99% OIL1, 1% OIL32
    std::vector<oil::FormatDescriptor> assignment;
    for (int i = 0; i < 100; i++) {
        assignment.push_back(i < 99 ? oil1 : oil32);
    }

    float avg = oil::FormatRegistry::compute_average_bpw(assignment);
    float expected = (99.0f * 1.0f + 1.0f * 32.0f) / 100.0f;
    TEST_CHECK_CLOSE(avg, expected, 0.1f, "Weighted 99/1 distribution");

    printf("  99/1 OIL1/OIL32 avg_bpw: %.2f (expected %.2f)\n", avg, expected);
}

void test_effective_bpw_two_mixes() {
    TEST_SUITE("Section 3: Two-mix effective BPW");
    const auto& two_mixes = oil::FormatRegistry::get_all_two_mixes();
    TEST_CHECK(two_mixes.size() > 0, "Two-mixes exist");

    for (const auto& m : two_mixes) {
        float expected = m.tier1_ratio *
            ([&]() {
                oil::FormatDescriptor f = oil::FormatRegistry::parse_format_name(
                    oil::format_name(oil::regformat_to_format(m.tier1_fmt)));
                return f.bpw;
            })() + m.tier2_ratio * ([&]() {
            oil::FormatDescriptor f = oil::FormatRegistry::parse_format_name(
                oil::format_name(oil::regformat_to_format(m.tier2_fmt)));
            return f.bpw;
        })();

        char msg[256];
        snprintf(msg, sizeof(msg), "%s eff_bpw=%.2f expected=%.2f",
                 m.name.c_str(), m.effective_bpw, expected);
        TEST_CHECK_CLOSE(m.effective_bpw, expected, 0.5f, msg);

        printf("  %s: eff_bpw=%.2f (expected ~%.2f)\n",
               m.name.c_str(), m.effective_bpw, expected);
    }
}

void test_format_assignment_optimizer() {
    TEST_SUITE("Section 3: format_assignment_optimizer");
    const auto& singles = oil::FormatRegistry::get_all_singles();
    TEST_CHECK(singles.size() == 15, "15 formats available");

    // Forced distribution at 2.0 BPW
    std::vector<float> data(1024);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data) v = dist(rng);

    auto assignment = oil::FormatRegistry::apply_forced_distribution(2.0f, 3, data.data(), 1024);
    TEST_CHECK(assignment.size() > 0, "Forced distribution returns formats");
    TEST_CHECK(assignment.size() <= 3, "Respects num_formats limit");

    float avg_bpw = oil::FormatRegistry::compute_average_bpw(assignment);
    printf("  2.0 BPW forced 3-format: avg_bpw=%.2f formats=%zu\n",
           avg_bpw, assignment.size());

    // Forced distribution at 8.0 BPW
    auto assignment8 = oil::FormatRegistry::apply_forced_distribution(8.0f, 2, data.data(), 1024);
    TEST_CHECK(assignment8.size() > 0, "8.0 BPW assignment returns formats");
    float avg_bpw8 = oil::FormatRegistry::compute_average_bpw(assignment8);
    printf("  8.0 BPW forced 2-format: avg_bpw=%.2f formats=%zu\n",
           avg_bpw8, assignment8.size());
}

void test_select_best_format() {
    TEST_SUITE("Section 3: select_best_format");
    std::mt19937 rng(555);
    std::vector<float> data(2048);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data) v = dist(rng);

    oil::FormatDescriptor best2 = oil::FormatRegistry::select_best_format(
        2.0f, data.data(), 2048);
    TEST_CHECK(!best2.name.empty(), "Best format for 2.0 BPW found");
    printf("  Best for 2.0 BPW: %s (bpw=%.1f)\n", best2.name.c_str(), best2.bpw);

    oil::FormatDescriptor best8 = oil::FormatRegistry::select_best_format(
        8.0f, data.data(), 2048);
    TEST_CHECK(!best8.name.empty(), "Best format for 8.0 BPW found");
    printf("  Best for 8.0 BPW: %s (bpw=%.1f)\n", best8.name.c_str(), best8.bpw);
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  SECTION 3: BPW 2.00 PROOF\n");
    printf("  SPARK_Q0, SPARK_SPARSE, SPARK_Q0_GRP = 2.0 BPW\n");
    printf("  compute_average_bpw, forced distribution, select_best\n");
    printf("═══════════════════════════════════════════════════════════\n");

    test_spark_q0_bpw();
    test_spark_sparse_bpw();
    test_spark_q0_grp_bpw();
    test_spark_sparse_grp_bpw();
    test_compute_average_bpw_single_format();
    test_compute_average_bpw_mixed();
    test_compute_average_bpw_empty();
    test_compute_average_bpw_all_2_00();
    test_compute_average_bpw_weighted_distribution();
    test_effective_bpw_two_mixes();
    test_format_assignment_optimizer();
    test_select_best_format();

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Section 3 complete.\n");
    printf("═══════════════════════════════════════════════════════════\n");

    TEST_REPORT();
    return 0;
}
