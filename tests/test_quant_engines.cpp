#include "oil/oil_engines.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/kernel.h"
#include "oil/codebook.h"
#include "oil/math.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdint>
#include "oil/test.h"

using namespace oil;
using namespace oil::engines;

static bool approx_equal(float a, float b, float eps = 1e-3f) {
    return std::abs(a - b) < eps;
}

static bool all_close(const float* a, const float* b, int64_t n, float eps = 1e-3f) {
    for (int64_t i = 0; i < n; i++)
        if (!approx_equal(a[i], b[i], eps)) return false;
    return true;
}

// OIL8 quantize/dequantize roundtrip
static void test_oil8_roundtrip() {
    TEST_SUITE("OIL8 quantize/dequantize");
    OIL8Engine oil8;
    std::vector<float> data(256);
    for (int i = 0; i < 256; i++)
        data[i] = (float)(i - 128) / 8.0f;
    oil8.train_codebook(data.data(), 256);

    float val = 5.0f;
    uint8_t idx = oil8.quantize(val);
    float deq = oil8.dequantize(idx);
    TEST_CHECK_CLOSE(val, deq, 1.0f, "OIL8 scalar roundtrip");

    // Tensor roundtrip
    int64_t N = 100;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = std::sin((float)i * 0.1f) * 8.0f;
    Tensor q = oil8.quantize_tensor(orig.data<float>(), N);
    Tensor dq = oil8.dequant_tensor(q.data<uint8_t>(), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 4.0f), "OIL8 tensor roundtrip");

    // Per-channel
    Tensor orig2d({4, 32});
    for (int64_t i = 0; i < 4 * 32; i++)
        orig2d.data<float>()[i] = (float)(i % 20);
    Tensor qc, scales;
    oil8.quantize_per_channel(orig2d, 0, qc, scales);
    Tensor dqc({4, 32});
    oil8.dequantize_per_channel(qc, scales, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 32, 5.0f), "OIL8 per-channel roundtrip");

    // quant_error
    float err = oil8.quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "OIL8 quant_error finite");
    TEST_CHECK(err >= 0, "OIL8 quant_error non-negative");

    // quant_snr
    float snr = oil8.quant_snr(orig, dq);
    TEST_CHECK(std::isfinite(snr), "OIL8 quant_snr finite");
}

// OIL4 quantize/dequantize roundtrip
static void test_oil4_roundtrip() {
    TEST_SUITE("OIL4 quantize/dequantize");
    OIL4Engine oil4;
    std::vector<float> train_data(64);
    for (int i = 0; i < 64; i++)
        train_data[i] = (float)(i - 32) / 2.0f;
    oil4.train_codebook(train_data.data(), 64);

    float val = 5.0f;
    uint8_t idx = oil4.quantize(val);
    float deq = oil4.dequantize(idx);
    TEST_CHECK_CLOSE(val, deq, 2.0f, "OIL4 scalar roundtrip");

    int64_t N = 64;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 32) / 4.0f;
    Tensor q = oil4.quantize_tensor(orig.data<float>(), N);
    Tensor dq = oil4.dequant_tensor(q.data<uint8_t>(), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 5.0f), "OIL4 tensor roundtrip");

    // Per-channel
    Tensor orig2d({4, 16});
    for (int64_t i = 0; i < 4 * 16; i++)
        orig2d.data<float>()[i] = (float)(i % 10);
    Tensor qc, scales;
    oil4.quantize_per_channel(orig2d, 0, qc, scales);
    Tensor dqc({4, 16});
    oil4.dequantize_per_channel(qc, scales, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 16, 5.0f), "OIL4 per-channel roundtrip");

    float err = oil4.quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "OIL4 quant_error finite");
}

// FP8 E4M3 roundtrip
static void test_fp8_e4m3() {
    TEST_SUITE("FP8 E4M3");
    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 4.0f, 8.0f, -8.0f, 3.14f, -3.14f, 0.125f};
    for (int i = 0; i < 12; i++) {
        uint8_t q = fp8_e4m3_quantize(vals[i]);
        float dq = fp8_e4m3_dequantize(q);
        TEST_CHECK_CLOSE(vals[i], dq, 0.5f, "E4M3 scalar roundtrip");
    }

    int64_t N = 50;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = std::sin((float)i) * 4.0f;
    Tensor q = fp8_e4m3_quantize_tensor(orig.data<float>(), N);
    Tensor dq = fp8_e4m3_dequant_tensor(q.data<uint8_t>(), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 1.0f), "E4M3 tensor roundtrip");

    // Per-channel
    Tensor orig2d({4, 16});
    for (int64_t i = 0; i < 4 * 16; i++)
        orig2d.data<float>()[i] = (float)(i % 8) - 4.0f;
    Tensor qc, sc;
    fp8_e4m3_quantize_per_channel(orig2d, 0, qc, sc);
    Tensor dqc({4, 16});
    fp8_e4m3_dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 16, 1.0f), "E4M3 per-channel roundtrip");

    float err = fp8_e4m3_quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "E4M3 quant_error finite");

    float snr = fp8_e4m3_quant_snr(orig, dq);
    TEST_CHECK(std::isfinite(snr), "E4M3 quant_snr finite");
}

// FP8 E5M2 roundtrip
static void test_fp8_e5m2() {
    TEST_SUITE("FP8 E5M2");
    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 4.0f, 8.0f, 16.0f, -16.0f, 32.0f, 64.0f, 128.0f};
    for (int i = 0; i < 12; i++) {
        uint8_t q = fp8_e5m2_quantize(vals[i]);
        float dq = fp8_e5m2_dequantize(q);
        TEST_CHECK_CLOSE(vals[i], dq, 2.0f, "E5M2 scalar roundtrip");
    }

    int64_t N = 50;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = std::sin((float)i) * 32.0f;
    Tensor q = fp8_e5m2_quantize_tensor(orig.data<float>(), N);
    Tensor dq = fp8_e5m2_dequant_tensor(q.data<uint8_t>(), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 4.0f), "E5M2 tensor roundtrip");

    // Per-channel
    Tensor orig2d({3, 20});
    for (int64_t i = 0; i < 3 * 20; i++)
        orig2d.data<float>()[i] = (float)(i % 10) - 5.0f;
    Tensor qc, sc;
    fp8_e5m2_quantize_per_channel(orig2d, 0, qc, sc);
    Tensor dqc({3, 20});
    fp8_e5m2_dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 3 * 20, 2.0f), "E5M2 per-channel roundtrip");

    float err = fp8_e5m2_quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "E5M2 quant_error finite");
}

// NF4 quantize/dequantize
static void test_nf4() {
    TEST_SUITE("NF4");
    float val = 0.5f;
    float scale = 1.0f;
    uint8_t idx = nf4_quantize(val, scale);
    float nf4_scalar_dq = nf4_dequantize(idx, scale);
    TEST_CHECK_CLOSE(val, nf4_scalar_dq, 0.5f, "NF4 scalar roundtrip");

    int64_t N = 64, BS = 32;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 32) / 8.0f;
    Tensor q = nf4_quantize_tensor(orig.data<float>(), N, BS);
    // Layout: [num_blocks float scales | N uint8 indices]
    int64_t nf4_blocks = (N + BS - 1) / BS;
    const float* nf4_scales = reinterpret_cast<const float*>(q.data<uint8_t>());
    const uint8_t* nf4_idx = q.data<uint8_t>() + nf4_blocks * (int64_t)sizeof(float);
    Tensor dq_t = nf4_dequant_tensor(nf4_idx, nf4_scales, N, BS);
    TEST_CHECK(all_close(orig.data<float>(), dq_t.data<float>(), N, 1.0f), "NF4 tensor roundtrip");

    // Per-channel
    Tensor orig2d({2, 32});
    for (int64_t i = 0; i < 2 * 32; i++)
        orig2d.data<float>()[i] = (float)(i % 8) - 4.0f;
    Tensor qc, sc;
    nf4_quantize_per_channel(orig2d, 0, BS, qc, sc);
    Tensor dqc({2, 32});
    nf4_dequantize_per_channel(qc, sc, BS, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 2 * 32, 1.0f), "NF4 per-channel roundtrip");

    float err = nf4_quant_error(orig, dq_t);
    TEST_CHECK(std::isfinite(err), "NF4 quant_error finite");
    float snr = nf4_quant_snr(orig, dq_t);
    TEST_CHECK(std::isfinite(snr), "NF4 quant_snr finite");
}

// AWQ quantize/dequantize
static void test_awq() {
    TEST_SUITE("AWQ");
    AWQQuantizer awq(32, 0.5f);
    Tensor weight({16, 32});
    for (int64_t i = 0; i < 16 * 32; i++)
        weight.data<float>()[i] = (float)(i % 16) - 8.0f;
    Tensor act({8, 32});
    for (int64_t i = 0; i < 8 * 32; i++)
        act.data<float>()[i] = std::sin((float)i * 0.1f);
    awq.compute_scales(weight, act);
    Tensor q = awq.quantize(weight);
    Tensor dq = awq.dequantize(q);
    TEST_CHECK(all_close(weight.data<float>(), dq.data<float>(), 16 * 32, 2.0f), "AWQ weight roundtrip");

    // Per-channel
    Tensor qc, sc;
    awq.quantize_per_channel(weight, 0, qc, sc);
    Tensor dqc({16, 32});
    awq.dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(weight.data<float>(), dqc.data<float>(), 16 * 32, 2.0f), "AWQ per-channel roundtrip");

    float err = awq.quant_error(weight, dq);
    TEST_CHECK(std::isfinite(err), "AWQ quant_error finite");
    float snr = awq.quant_snr(weight, dq);
    TEST_CHECK(std::isfinite(snr), "AWQ quant_snr finite");
}

// GPTQ quantize/dequantize
static void test_gptq() {
    TEST_SUITE("GPTQ");
    GPTQQuantizer gptq(32, 4);
    Tensor weight({8, 16});
    for (int64_t i = 0; i < 8 * 16; i++)
        weight.data<float>()[i] = (float)(i % 8) - 4.0f;
    Tensor hessian({16, 16});
    for (int64_t i = 0; i < 16; i++)
        for (int64_t j = 0; j < 16; j++)
            hessian.data<float>()[i * 16 + j] = (i == j) ? 2.0f : 0.1f;
    Tensor q = gptq.quantize(weight, hessian);
    Tensor dq = gptq.dequantize(q);
    TEST_CHECK(all_close(weight.data<float>(), dq.data<float>(), 8 * 16, 4.0f), "GPTQ weight roundtrip");

    // Per-channel
    Tensor qc, sc;
    gptq.quantize_per_channel(weight, 0, qc, sc);
    Tensor dqc({8, 16});
    gptq.dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(weight.data<float>(), dqc.data<float>(), 8 * 16, 2.0f), "GPTQ per-channel roundtrip");

    float err = gptq.quant_error(weight, dq);
    TEST_CHECK(std::isfinite(err), "GPTQ quant_error finite");
}

// SPARK quantize/dequantize
static void test_spark() {
    TEST_SUITE("Spark");
    SparkEngine te(32);
    int64_t N = 64;
    Tensor orig({N});
    // Normalized weights in [-1, 1] — typical for SPARK quantization
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 32) / 32.0f;
    Tensor q = te.quantize(orig);
    Tensor dq = te.dequantize(q, Tensor({1}), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 0.7f), "Spark tensor roundtrip");

    // Per-channel — wider range with per-channel scaling
    Tensor orig2d({4, 16});
    for (int64_t i = 0; i < 4 * 16; i++)
        orig2d.data<float>()[i] = (float)(i % 8) - 4.0f;
    Tensor qc, sc;
    te.quantize_per_channel(orig2d, 0, qc, sc);
    Tensor dqc({4, 16});
    te.dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 16, 3.0f), "Spark per-channel roundtrip");

    float err = te.quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "Spark quant_error finite");
}

// OIL1 quantize/dequantize
static void test_oil1() {
    TEST_SUITE("Oil1");
    Oil1Engine be;
    int64_t N = 64;
    Tensor orig({N});
    // Normalized weights in [-1, 1] — typical for OIL1 quantization
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 32) / 32.0f;
    Tensor q = be.quantize(orig);
    float scale = 1.0f;
    Tensor dq = be.dequantize(q, scale, N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 1.1f), "Oil1 tensor roundtrip");

    // Per-channel
    Tensor orig2d({4, 16});
    for (int64_t i = 0; i < 4 * 16; i++)
        orig2d.data<float>()[i] = (float)(i % 8) - 4.0f;
    Tensor qc, sc;
    be.quantize_per_channel(orig2d, 0, qc, sc);
    Tensor dqc({4, 16});
    be.dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 16, 4.1f), "Oil1 per-channel roundtrip");

    float err = be.quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "Oil1 quant_error finite");
}

// quant_gemm fused operation
static void test_quant_gemm() {
    TEST_SUITE("Quant GEMM fused");
    int64_t M = 2, N = 4, K = 8;

    // FP8 E4M3 quant_gemm
    Tensor a({M, K});
    std::vector<uint8_t> b_q(K * N);
    for (int64_t i = 0; i < M * K; i++)
        a.data<float>()[i] = std::sin((float)i) * 2.0f;
    for (int64_t i = 0; i < K * N; i++)
        b_q[i] = fp8_e4m3_quantize(std::cos((float)i) * 2.0f);
    Tensor c_e4m3 = fp8_e4m3_quant_gemm(a, b_q.data(), M, N, K);
    TEST_CHECK(c_e4m3.numel() == M * N, "E4M3 quant_gemm output shape");
    for (int64_t i = 0; i < c_e4m3.numel(); i++)
        TEST_CHECK(std::isfinite(c_e4m3.data<float>()[i]), "E4M3 quant_gemm finite");

    // FP8 E5M2 quant_gemm
    Tensor c_e5m2 = fp8_e5m2_quant_gemm(a, b_q.data(), M, N, K);
    TEST_CHECK(c_e5m2.numel() == M * N, "E5M2 quant_gemm output shape");
    for (int64_t i = 0; i < c_e5m2.numel(); i++)
        TEST_CHECK(std::isfinite(c_e5m2.data<float>()[i]), "E5M2 quant_gemm finite");

    // OIL8 quant_gemm
    OIL8Engine oil8;
    std::vector<float> train(256);
    for (int i = 0; i < 256; i++) train[i] = (float)(i - 128) / 16.0f;
    oil8.train_codebook(train.data(), 256);
    std::vector<uint8_t> b_idx(K * N);
    for (int64_t i = 0; i < K * N; i++)
        b_idx[i] = oil8.quantize((float)(i % 10));
    Tensor c_oil8 = oil8.quant_gemm(a, b_idx.data(), M, N, K);
    TEST_CHECK(c_oil8.numel() == M * N, "OIL8 quant_gemm output shape");
    for (int64_t i = 0; i < c_oil8.numel(); i++)
        TEST_CHECK(std::isfinite(c_oil8.data<float>()[i]), "OIL8 quant_gemm finite");
}

// quant_error metrics
static void test_quant_error_metrics() {
    TEST_SUITE("Quant Error Metrics");
    int64_t N = 32;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 16) / 4.0f;
    Tensor perfect = orig.clone();

    float err = compute_quant_error(orig, perfect);
    TEST_CHECK_CLOSE(err, 0.0f, 1e-6f, "quant_error zero for identical");
    float mse = compute_quant_mse(orig, perfect);
    TEST_CHECK_CLOSE(mse, 0.0f, 1e-6f, "quant_mse zero for identical");

    // Non-zero error
    Tensor bad = orig.clone();
    bad.fill(0.0f);
    float err_bad = compute_quant_error(orig, bad);
    TEST_CHECK(err_bad > 0, "quant_error positive for mismatch");
    float mse_bad = compute_quant_mse(orig, bad);
    TEST_CHECK(mse_bad > 0, "quant_mse positive for mismatch");

    // SNR computation
    float snr = compute_quant_snr(orig, perfect);
    TEST_CHECK(std::isfinite(snr), "quant_snr finite for identical");
}

// Per-block OIL8 vs Q8_0: prove OIL8 per-block beats uniform Q8_0
static void test_perblock_oil8_beats_q8() {
    TEST_SUITE("Per-block OIL8 vs Q8_0");
    const int64_t N = 100320;
    const int64_t BS = 32;

    std::vector<float> w(N);
    {
        unsigned seed = 42;
        auto rng = [&]() { seed = seed * 1103515245 + 12345; return (float)((seed >> 16) & 0x7FFF) / 32768.0f; };
        for (int64_t i = 0; i < N; i++) {
            float u1 = rng() * 0.999f + 0.001f;
            float u2 = rng();
            float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
            w[i] = z * 0.02f;
        }
    }

    // Q8_0 per-block uniform
    double q8_sq = 0.0;
    for (int64_t b = 0; b < N / BS; b++) {
        float amax = 0.0f;
        for (int64_t i = 0; i < BS; i++)
            amax = (std::max)(amax, std::abs(w[b * BS + i]));
        float scale = amax / 127.0f;
        if (scale < 1e-10f) scale = 1.0f;
        for (int64_t i = 0; i < BS; i++) {
            float q = std::round(w[b * BS + i] / scale);
            q = (std::max)(-128.0f, (std::min)(127.0f, q));
            float dq = q * scale;
            double d = (double)w[b * BS + i] - (double)dq;
            q8_sq += d * d;
        }
    }
    double q8_mse = q8_sq / (double)N;

    // OIL8 per-block k-means
    OIL8Engine oil8;
    oil8.train_codebook_per_block(w.data(), N, BS, 30);

    std::vector<uint8_t> indices(N);
    std::vector<float> scales(N / BS + 1);
    oil8.quantize_per_block(w.data(), N, BS, indices.data(), scales.data());

    std::vector<float> dq(N);
    oil8.dequantize_per_block(indices.data(), scales.data(), N, BS, dq.data());

    double oil8_sq = 0.0;
    for (int64_t i = 0; i < N; i++) {
        double d = (double)w[i] - (double)dq[i];
        oil8_sq += d * d;
    }
    double oil8_mse = oil8_sq / (double)N;

    printf("  Q8_0  (uniform per-block) MSE: %.6e\n", q8_mse);
    printf("  OIL8  (k-means per-block) MSE: %.6e\n", oil8_mse);
    if (oil8_mse < q8_mse) {
        printf("  >>> OIL8 BEATS Q8_0 by %.2fx <<<\n", q8_mse / oil8_mse);
    } else {
        printf("  Q8_0 wins by %.2fx\n", oil8_mse / q8_mse);
    }
    TEST_CHECK(oil8_mse < q8_mse, "OIL8 per-block beats Q8_0 uniform");
}

// Per-block OIL4 vs Q4_0: prove OIL4 per-block beats uniform Q4_0
static void test_perblock_oil4_beats_q4() {
    TEST_SUITE("Per-block OIL4 vs Q4_0");
    const int64_t N = 100320;
    const int64_t BS = 32;

    std::vector<float> w(N);
    {
        unsigned seed = 42;
        auto rng = [&]() { seed = seed * 1103515245 + 12345; return (float)((seed >> 16) & 0x7FFF) / 32768.0f; };
        for (int64_t i = 0; i < N; i++) {
            float u1 = rng() * 0.999f + 0.001f;
            float u2 = rng();
            float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
            w[i] = z * 0.02f;
        }
    }

    // Q4_0 per-block uniform
    double q4_sq = 0.0;
    for (int64_t b = 0; b < N / BS; b++) {
        float bmin = w[b * BS], bmax = w[b * BS];
        for (int64_t i = 1; i < BS; i++) {
            bmin = (std::min)(bmin, w[b * BS + i]);
            bmax = (std::max)(bmax, w[b * BS + i]);
        }
        float range = bmax - bmin;
        float scale = range / 15.0f;
        if (scale < 1e-10f) scale = 1.0f;
        for (int64_t i = 0; i < BS; i++) {
            float q = std::round((w[b * BS + i] - bmin) / scale);
            q = (std::max)(0.0f, (std::min)(15.0f, q));
            float dq = q * scale + bmin;
            double d = (double)w[b * BS + i] - (double)dq;
            q4_sq += d * d;
        }
    }
    double q4_mse = q4_sq / (double)N;

    // OIL4 per-block k-means
    OIL4Engine oil4;
    oil4.train_codebook_per_block(w.data(), N, BS, 30);

    std::vector<uint8_t> indices(N);
    std::vector<float> scales((N / BS + 1) * 2);
    oil4.quantize_per_block(w.data(), N, BS, indices.data(), scales.data());

    std::vector<float> dq(N);
    oil4.dequantize_per_block(indices.data(), scales.data(), N, BS, dq.data());

    double oil4_sq = 0.0;
    for (int64_t i = 0; i < N; i++) {
        double d = (double)w[i] - (double)dq[i];
        oil4_sq += d * d;
    }
    double oil4_mse = oil4_sq / (double)N;

    printf("  Q4_0  (uniform per-block) MSE: %.6e\n", q4_mse);
    printf("  OIL4  (k-means per-block) MSE: %.6e\n", oil4_mse);
    if (oil4_mse < q4_mse) {
        printf("  >>> OIL4 BEATS Q4_0 by %.2fx <<<\n", q4_mse / oil4_mse);
    } else {
        printf("  Q4_0 wins by %.2fx\n", oil4_mse / q4_mse);
    }
    TEST_CHECK(oil4_mse < q4_mse, "OIL4 per-block beats Q4_0 uniform");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — Quantization Engine Test Suite\n");
    printf("============================================\n");

    test_oil8_roundtrip();
    test_oil4_roundtrip();
    test_fp8_e4m3();
    test_fp8_e5m2();
    test_nf4();
    test_awq();
    test_gptq();
    test_spark();
    test_oil1();
    test_quant_gemm();
    test_quant_error_metrics();
    test_perblock_oil8_beats_q8();
    test_perblock_oil4_beats_q4();

    printf("\n============================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
