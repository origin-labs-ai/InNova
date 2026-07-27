#include "oil/test.h"
#include "oil/format_registry.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

static constexpr int FORMAT_COUNT = 15;
static constexpr int NUM_LOSSLESS = 8;
static constexpr int NUM_LOSSY = 7;
static constexpr int SEED = 42;

static const char* EXPECTED_NAMES[FORMAT_COUNT] = {
    "OIL1", "OIL2", "OIL4", "OIL8", "OIL16", "OIL32",
    "OIL1_GRP", "OIL2_GRP", "OIL4_GRP", "OIL8_GRP", "OIL16_GRP",
    "SPARK_SPARSE", "SPARK_SPARSE_GRP", "SPARK_Q0", "SPARK_Q0_GRP"
};

static constexpr float EXPECTED_BPW[FORMAT_COUNT] = {
    1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f,
    1.0f, 2.0f, 4.0f, 8.0f, 16.0f,
    1.5f, 2.0f, 1.5f, 1.5f
};

static constexpr bool EXPECTED_LOSSLESS[FORMAT_COUNT] = {
    false, false, false, false, false, true,
    true, true, true, true, true,
    false, true, false, true
};

void test_registry_count() {
    TEST_SUITE("Registry Completeness");
    const auto& singles = oil::FormatRegistry::get_all_singles();
    TEST_CHECK((int)singles.size() == FORMAT_COUNT, "Registry must have exactly 15 formats");
    printf("  Format count: %d (expected %d)\n", (int)singles.size(), FORMAT_COUNT);
}

void test_registry_names() {
    TEST_SUITE("Registry Names");
    const auto& singles = oil::FormatRegistry::get_all_singles();
    for (int i = 0; i < FORMAT_COUNT; i++) {
        bool found = false;
        for (const auto& s : singles) {
            if (s.name == EXPECTED_NAMES[i]) { found = true; break; }
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "Format '%s' must exist in registry", EXPECTED_NAMES[i]);
        TEST_CHECK(found, msg);
    }
}

void test_no_extras() {
    TEST_SUITE("No Extra Formats");
    const auto& singles = oil::FormatRegistry::get_all_singles();
    for (const auto& s : singles) {
        bool expected = false;
        for (int i = 0; i < FORMAT_COUNT; i++) {
            if (s.name == EXPECTED_NAMES[i]) { expected = true; break; }
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "Unexpected format '%s' in registry", s.name.c_str());
        TEST_CHECK(expected, msg);
    }
}

void test_lossless_flags() {
    TEST_SUITE("Lossless Flags");
    const auto& singles = oil::FormatRegistry::get_all_singles();
    int lossless_count = 0;
    for (int i = 0; i < FORMAT_COUNT; i++) {
        for (const auto& s : singles) {
            if (s.name == EXPECTED_NAMES[i]) {
                char msg[128];
                snprintf(msg, sizeof(msg), "%s lossless=%s (expected %s)",
                         s.name.c_str(), s.lossless ? "true" : "false",
                         EXPECTED_LOSSLESS[i] ? "true" : "false");
                TEST_CHECK(s.lossless == EXPECTED_LOSSLESS[i], msg);
                if (s.lossless) lossless_count++;
                break;
            }
        }
    }
    TEST_CHECK(lossless_count == NUM_LOSSLESS, "Exactly 8 lossless formats");
    printf("  Lossless count: %d (expected %d)\n", lossless_count, NUM_LOSSLESS);
}

void test_bpw_values() {
    TEST_SUITE("BPW Values");
    const auto& singles = oil::FormatRegistry::get_all_singles();
    for (int i = 0; i < FORMAT_COUNT; i++) {
        for (const auto& s : singles) {
            if (s.name == EXPECTED_NAMES[i]) {
                char msg[128];
                snprintf(msg, sizeof(msg), "%s BPW=%.1f (expected %.1f)",
                         s.name.c_str(), s.bpw, EXPECTED_BPW[i]);
                TEST_CHECK_CLOSE(s.bpw, EXPECTED_BPW[i], 0.1f, msg);
                break;
            }
        }
    }
}

void test_mse_populated() {
    TEST_SUITE("MSE Populated");
    const auto& singles = oil::FormatRegistry::get_all_singles();
    int mse_count = 0;
    for (const auto& s : singles) {
        if (s.est_mse > 0.0f || s.lossless) mse_count++;
    }
    TEST_CHECK(mse_count == FORMAT_COUNT, "All formats have MSE or are lossless");
}

void test_format_table() {
    TEST_SUITE("Format Table");
    std::string table = oil::FormatRegistry::get_format_table();
    TEST_CHECK(table.size() > 100, "Format table is non-trivial");
    TEST_CHECK(table.find("OIL1") != std::string::npos, "Table contains OIL1");
    TEST_CHECK(table.find("SPARK_Q0") != std::string::npos, "Table contains SPARK_Q0");
    TEST_CHECK(table.find("OIL32") != std::string::npos, "Table contains OIL32");
    printf("  Table length: %zu chars\n", table.size());
}

void test_parse_format_names() {
    TEST_SUITE("Parse Format Names");
    for (int i = 0; i < FORMAT_COUNT; i++) {
        oil::FormatDescriptor fd = oil::FormatRegistry::parse_format_name(EXPECTED_NAMES[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "parse_format_name('%s') returns valid descriptor", EXPECTED_NAMES[i]);
        TEST_CHECK(!fd.name.empty(), msg);
    }
    oil::FormatDescriptor unknown = oil::FormatRegistry::parse_format_name("NONEXISTENT");
    TEST_CHECK(unknown.name.empty(), "parse_format_name('NONEXISTENT') returns empty");
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  FORMAT REGISTRY COMPLETENESS TEST\n");
    printf("  Expected: 15 OIL/SPARK formats, no Ternary, no Binary\n");
    printf("═══════════════════════════════════════════════════════════\n");

    test_registry_count();
    test_registry_names();
    test_no_extras();
    test_lossless_flags();
    test_bpw_values();
    test_mse_populated();
    test_format_table();
    test_parse_format_names();

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  15 formats. All tests complete.\n");
    printf("═══════════════════════════════════════════════════════════\n");

    TEST_REPORT();
    return 0;
}
