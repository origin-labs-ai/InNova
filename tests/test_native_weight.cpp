#include "oil/native_weight.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include "oil/test.h"

using namespace oil;

// Test only NativeOILWeightStore — no model, no autograd, no trainer
int main() {
    printf("=== Native OIL Weight Store Tests ===\n\n");

    // Test basic operations
    printf("--- Test 1: Basic create/convert/dequantize ---\n");
    const size_t N = 256;
    native::NativeOILWeightStore store(N, 128);
    TEST_CHECK(store.size() == N, "size=256");
    TEST_CHECK(store.block_size() == 128, "block_size=128");
    TEST_CHECK(store.num_blocks() == 2, "num_blocks=2");

    std::vector<float> src(N);
    std::vector<float> dst(N);
    for (size_t i = 0; i < N; i++) src[i] = (float)((int)i % 3 - 1);
    store.convert_from_fp32(src.data());
    store.dequantize(dst.data());
    float s0 = store.get_scale(0);
    TEST_CHECK(std::abs(dst[0] + s0) < 1e-5f, "deq(-1) ≈ -scale");
    TEST_CHECK(std::abs(dst[1]) < 1e-5f, "deq(0) ≈ 0");
    TEST_CHECK(std::abs(dst[2] - s0) < 1e-5f, "deq(+1) ≈ +scale");

    printf("  scale=%f deq[0]=%f deq[1]=%f deq[2]=%f\n", s0, dst[0], dst[1], dst[2]);

    // Test CID allocation — use enough blocks to see all 3 formats
    // With block_size=128 and frac_oil1=0.05, need N >= 2560 for a full OIL1 block
    printf("\n--- Test 2: CID allocation ---\n");
    const size_t N2 = 2560;  // 20 blocks of 128
    native::NativeOILWeightStore store_cid(N2, 128);
    std::vector<float> sens2(N2);
    // Strictly decreasing sensitivity for deterministic sort order
    for (size_t i = 0; i < N2; i++) {
        if (i < 128) sens2[i] = 1000.0f - (float)i;                    // top 5%
        else if (i < N2 - 128) sens2[i] = 500.0f - (float)(i - 128) * 0.1f;  // middle 90%
        else sens2[i] = 0.001f - (float)i * 1e-9f;                      // bottom 5%
    }
    store_cid.reallocate_by_sensitivity(sens2.data(), 0.05f, 0.90f);
    printf("  Block 0 (highest sens) format: %d (expect 0=OIL8)\n", (int)store_cid.get_format(0));
    printf("  Block 10 (middle) format: %d (expect 1=OIL1)\n", (int)store_cid.get_format(1280));
    printf("  Block 19 (lowest sens) format: %d (expect 2=OIL4)\n", (int)store_cid.get_format(2559));
    TEST_CHECK(store_cid.get_format(0) == native::NativeFormat::OIL8, "top block → OIL8");
    TEST_CHECK(store_cid.get_format(1280) == native::NativeFormat::OIL1, "middle block → OIL1");
    TEST_CHECK(store_cid.get_format(2559) == native::NativeFormat::OIL4, "bottom block → OIL4");

    // Test two-timescale update with dead zone (Theorem 5d.3)
    printf("\n--- Test 3: Apply OIL update ---\n");
    native::NativeOILWeightStore store2(64, 64);
    std::vector<float> src2(64, 1.0f);
    store2.convert_from_fp32(src2.data());
    printf("  Initial: idx=%d scale=%f weight=%f\n",
           store2.get_index(0), store2.get_scale(0), store2.get_weight(0));

    // Small gradient — should stay in dead zone (|lr*grad| < scale*0.5)
    std::vector<float> small_grad(64, 0.01f);
    store2.apply_oil_update(small_grad.data(), 0.01f, 0.01f);
    printf("  After small gradient: idx=%d scale=%f weight=%f\n",
           store2.get_index(0), store2.get_scale(0), store2.get_weight(0));
    TEST_CHECK(store2.get_index(0) == 2, "index unchanged (dead zone for SPARK +1)");

    // Large POSITIVE gradient — pushes weight DOWN, should cross dead zone
    // Use lr_scale=0 (scale handled by Adam in trainer) to isolate index update
    // w_virtual = 1.0 - 0.1 * 100 = -9.0 → norm = -9/scale < -0.5 → idx=0
    std::vector<float> big_grad(64, 100.0f);
    store2.apply_oil_update(big_grad.data(), 0.0f, 0.1f);
    printf("  After large +gradient: idx=%d scale=%f weight=%f\n",
           store2.get_index(0), store2.get_scale(0), store2.get_weight(0));
    // Index should flip from 2 (+1) to 0 (-1) since virtual weight went negative
    bool idx_changed = (store2.get_index(0) != 2);
    TEST_CHECK(idx_changed, "index changed (exceeded dead zone)");

    // Test formats_data and indices_data accessors (for checkpoint I/O)
    printf("\n--- Test 4: Raw data access ---\n");
    const uint8_t* idx_ptr = store2.indices_data();
    const float* sc_ptr = store2.block_scales_data();
    const native::NativeFormat* fmt_ptr = store2.formats_data();
    TEST_CHECK(idx_ptr != nullptr, "indices_data accessible");
    TEST_CHECK(sc_ptr != nullptr, "block_scales_data accessible");
    TEST_CHECK(fmt_ptr != nullptr, "formats_data accessible");
    TEST_CHECK(idx_ptr[0] == store2.get_index(0), "indices_data[0] matches get_index(0)");

    return TEST_REPORT() > 0 ? 1 : 0;
}
