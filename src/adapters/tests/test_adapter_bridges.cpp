// ============================================================================
// test_adapter_bridges.cpp — Smoke tests for all ADAPTER-EDITION bridges
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/ptq_bridge.h"
#include "adapters/gguf_bridge.h"
#include "adapters/safetensors_bridge.h"

#include <cstdio>
#include <cstdlib>
// ctime removed — using <random> instead
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include "quant/quant_format.h"
#include "quant/block_codec.h"
#include "quant/tensor.h"

using namespace quant::adapters;
using namespace quant;

static int g_failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); g_failures++; }
    else      { std::fprintf(stdout, "PASS: %s\n", msg); }
}

static void test_fp_conversions() {
    std::fprintf(stdout, "\n=== FP Conversion Tests ===\n");

    float f16_1 = fp16_to_float(0x3C00);
    check(std::fabs(f16_1 - 1.0f) < 0.01f, "FP16 1.0");

    float f16_neg = fp16_to_float(0xBC00);
    check(std::fabs(f16_neg - (-1.0f)) < 0.01f, "FP16 -1.0");

    float f16_half = fp16_to_float(0x3800);
    check(std::fabs(f16_half - 0.5f) < 0.01f, "FP16 0.5");

    float bf16_1 = bf16_to_float(0x3F80);
    check(std::fabs(bf16_1 - 1.0f) < 0.01f, "BF16 1.0");

    float fp8_1 = fp8_e4m3_to_float(0x38);
    check(std::fabs(fp8_1 - 1.0f) < 0.01f, "FP8 E4M3 1.0");

    float fp8_1e5m2 = fp8_e5m2_to_float(0x3C);
    check(std::fabs(fp8_1e5m2 - 1.0f) < 0.01f, "FP8 E5M2 1.0");
}

static void test_format_detection() {
    std::fprintf(stdout, "\n=== Format Detection Tests ===\n");

    check(detect_format("test.bin") == ExternalFormat::RAW_FP32, ".bin = RAW_FP32");
    check(detect_format("test.fp32") == ExternalFormat::RAW_FP32, ".fp32 = RAW_FP32");
    check(detect_format("test.fp16") == ExternalFormat::RAW_FP16, ".fp16 = RAW_FP16");
    check(detect_format("test.gguf") == ExternalFormat::GGUF, ".gguf = GGUF");
    check(detect_format("test.safetensors") == ExternalFormat::SAFETENSORS, ".safetensors = SAFETENSORS");
    check(detect_format("test.quant") == ExternalFormat::QUANT, ".quant = QUANT");
    check(detect_format("test.fp8e4m3") == ExternalFormat::RAW_FP8_E4M3, ".fp8e4m3 = FP8");
    check(detect_format("test.unknown") == ExternalFormat::UNKNOWN, ".unknown = UNKNOWN");
}

static void test_mixed_write() {
    std::fprintf(stdout, "\n=== Mixed-Precision Write Tests ===\n");

    BridgeConfig cfg;
    cfg.target_bpw = 2.0f;
    cfg.block_size = 256;
    cfg.output_path = "test_adapter.quant";

    std::vector<AdapterTensor> tensors;

    AdapterTensor t1;
    t1.name = "layer0.weight";
    t1.shape = {64, 64};
    t1.data.resize(64 * 64);
    static thread_local std::mt19937 rng(42);
    for (auto& v : t1.data) v = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * 0.1f;
    tensors.push_back(std::move(t1));

    AdapterTensor t2;
    t2.name = "layer1.weight";
    t2.shape = {64, 64};
    t2.data.resize(64 * 64);
    for (auto& v : t2.data) v = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * 0.1f;
    tensors.push_back(std::move(t2));

    bool ok = write_quant_mixed(tensors, cfg);
    check(ok, "write_quant_mixed returns true");

    float bpw = estimate_mixed_bpw(64 * 64, 256, 2.0f);
    check(bpw > 1.0f && bpw < 4.0f, "estimated BPW in valid range");
}

static void test_quad_budget_at_scale() {
    std::fprintf(stdout, "\n=== QUAD_MIX Budget Tests ===\n");

    constexpr int64_t N = 1 << 20; // 1M weights, 4096 blocks of 256
    std::vector<float> data((size_t)N);
    std::mt19937 rng(20260802);
    for (int64_t g = 0; g < N / 1024; ++g) {
        const float scale = 0.20f + 0.15f * (float)(g + 1);
        std::normal_distribution<float> dist(0.0f, scale);
        for (int64_t k = 0; k < 1024; ++k) data[(size_t)(g * 1024 + k)] = dist(rng);
    }

    const MixDescriptor* mix = nullptr;
    for (const auto& m : FormatRegistry::get_all_four_mixes())
        if (m.name == "QUAD_QUANT2_QUANT4_QUANT8_QUANT16") mix = &m;
    check(mix != nullptr, "QUAD_QUANT2_QUANT4_QUANT8_QUANT16 registered");
    if (!mix) return;

    const std::vector<Format> fmts =
        allocate_tensor_formats("layer0.weight", N, data.data(), 256, Format::QUANT2, mix);

    size_t total_bytes = 0;
    bool budget_ok = true;
    for (int b = 0; b < (int)fmts.size(); ++b) {
        const Format bfmt = fmts[(size_t)b];
        const int start = b * 256;
        const int n = (int)std::min<int64_t>(256, N - start);
        std::vector<uint8_t> indices, codebook;
        quantize_block(bfmt, data.data() + start, n, indices, codebook);
        const size_t stored = indices.size() + codebook.size();
        const size_t cap = block_claimed_bytes(bfmt, (uint32_t)n);
        if (stored > cap) budget_ok = false;
        total_bytes += stored;
    }
    check(budget_ok, "every QUAD block fits its claimed byte budget");

    const double expected = 2.92 * (double)N / 8.0;
    const double actual = (double)total_bytes;
    char message[256];
    std::snprintf(message, sizeof(message),
                  "QUAD file bytes %.0f vs claimed %.0f (%.2f%% off)",
                  actual, expected, 100.0 * (actual - expected) / expected);
    check(std::fabs(actual - expected) / expected < 0.01, message);
    std::fprintf(stdout, "QUAD_QUANT2_QUANT4_QUANT8_QUANT16 @ 1M weights: %zu bytes (claim 2.92 BPW -> %.0f)\n",
                 total_bytes, expected);
}

static void test_raw_load() {
    std::fprintf(stdout, "\n=== Raw Load Tests ===\n");

    std::vector<float> data = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
    std::string path = "test_adapter_raw.fp32";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    f.close();

    AdapterTensor tensor = load_raw_blob(path, ExternalFormat::RAW_FP32);
    check(!tensor.data.empty(), "load_raw_blob loads FP32");
    if (!tensor.data.empty()) {
        check(tensor.data.size() == 6, "raw load gets correct count");
    }
}

int main() {
    std::fprintf(stdout, "========================================\n");
    std::fprintf(stdout, "  ADAPTER-EDITION SMOKE TESTS\n");
    std::fprintf(stdout, "========================================\n");

    test_fp_conversions();
    test_format_detection();
    test_mixed_write();
    test_quad_budget_at_scale();
    test_raw_load();

    std::fprintf(stdout, "\n========================================\n");
    if (g_failures == 0)
        std::fprintf(stdout, "  ALL TESTS PASSED\n");
    else
        std::fprintf(stdout, "  %d TEST(S) FAILED\n", g_failures);
    std::fprintf(stdout, "========================================\n");

    return g_failures;
}
