#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/math.h"
#include "oil/math_tiled.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>
#include <functional>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// BenchResult – one benchmark measurement
// ---------------------------------------------------------------------------
struct BenchResult {
    std::string name;
    int M, N, K;
    double time_us;
    double gflops;
    double ops_per_sec;
};

// ---------------------------------------------------------------------------
// Generic GEMM bench (raw pointer)
// ---------------------------------------------------------------------------
static BenchResult bench_gemm_raw(const std::string& name,
                                   void (*gemm_fn)(const float*, const float*, float*, int, int, int),
                                   int M, int N, int K) {
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    int warmup = 5;
    for (int i = 0; i < warmup; i++)
        gemm_fn(A.data(), B.data(), C.data(), M, N, K);

    int iters = 1000;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        gemm_fn(A.data(), B.data(), C.data(), M, N, K);
    double dt = (now_sec() - t0) / iters;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    return {name, M, N, K, dt * 1e6, flops / dt * 1e-9, 0.0};
}

// ---------------------------------------------------------------------------
// Tiled GEMM bench
// ---------------------------------------------------------------------------
static BenchResult bench_gemm_tiled(const std::string& name, int M, int N, int K) {
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    int warmup = 5;
    for (int i = 0; i < warmup; i++)
        oil::math::gemm_tiled(1.0f, A.data(), K, B.data(), N,
                              0.0f, C.data(), N, M, N, K);

    int iters = 1000;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        oil::math::gemm_tiled(1.0f, A.data(), K, B.data(), N,
                              0.0f, C.data(), N, M, N, K);
    double dt = (now_sec() - t0) / iters;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    return {name, M, N, K, dt * 1e6, flops / dt * 1e-9, 0.0};
}

// ---------------------------------------------------------------------------
// bench_gemm_comparison: scalar vs AVX2 vs AVX512 vs tiled at multiple sizes
// ---------------------------------------------------------------------------
static void bench_gemm_comparison() {
    std::cout << "\n=== GEMM Kernel Comparison ===" << std::endl;
    std::cout << std::left
              << std::setw(18) << "Variant"
              << std::setw(14) << "Size"
              << std::setw(14) << "Time (us)"
              << std::setw(12) << "GFLOPS"
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    std::vector<int> sizes = {128, 256, 512};

    for (int sz : sizes) {
        std::vector<BenchResult> results;

        auto r_scalar = bench_gemm_raw("scalar",
            oil::kernel::scalar_gemm, sz, sz, sz);
        results.push_back(r_scalar);

#if defined(OIL_AVX2)
        auto r_avx2 = bench_gemm_raw("avx2",
            oil::kernel::avx2_gemm, sz, sz, sz);
        results.push_back(r_avx2);

        auto r_avx2t = bench_gemm_raw("avx2_tiled",
            oil::kernel::avx2_tiled_gemm, sz, sz, sz);
        results.push_back(r_avx2t);
#endif

        auto r_tiled = bench_gemm_tiled("tiled64", sz, sz, sz);
        results.push_back(r_tiled);

        double base = results[0].time_us;
        for (auto& r : results) {
            std::cout << std::left
                      << std::setw(18) << r.name
                      << std::setw(14) << (std::to_string(r.M) + "x" + std::to_string(r.N) + "x" + std::to_string(r.K))
                      << std::setw(14) << std::fixed << std::setprecision(2) << r.time_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.gflops
                      << std::setw(10) << std::fixed << std::setprecision(2) << (base / r.time_us) << "x"
                      << std::endl;
        }
        std::cout << std::endl;
    }
}

// ---------------------------------------------------------------------------
// bench_dot_product: scalar vs AVX2 vectorized dot product
// ---------------------------------------------------------------------------
static void bench_dot_product() {
    std::cout << "\n=== Dot Product ===" << std::endl;

    std::vector<int64_t> sizes = {1024, 4096, 16384, 65536};

    for (int64_t n : sizes) {
        std::vector<float> a(n, 1.0f);
        std::vector<float> b(n, 2.0f);

        oil::Tensor ta(oil::Shape{n}, oil::DType::F32);
        oil::Tensor tb(oil::Shape{n}, oil::DType::F32);
        std::memcpy(ta.data<float>(), a.data(), n * sizeof(float));
        std::memcpy(tb.data<float>(), b.data(), n * sizeof(float));

        int warmup = 5;
        for (int i = 0; i < warmup; i++) oil::math::dot(ta, tb);

        int iters = 5000;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) oil::math::dot(ta, tb);
        double dt = (now_sec() - t0) / iters;

        double ops = 2.0 * n;
        std::cout << "  n=" << std::setw(6) << n
                  << "  time: " << std::fixed << std::setprecision(2) << (dt * 1e6) << " us"
                  << "  GFLOPS: " << std::fixed << std::setprecision(2) << (ops / dt * 1e-9)
                  << "  ops/s: " << std::scientific << std::setprecision(2) << (ops / dt) << std::endl;
    }
}

// ---------------------------------------------------------------------------
// bench_gemm_transpose: AB vs AtB vs ABt vs AtBt variants
// ---------------------------------------------------------------------------
static void bench_gemm_transpose() {
    std::cout << "\n=== GEMM Transpose Variants (256x256x256) ===" << std::endl;

    const int M = 256, N = 256, K = 256;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    for (int64_t i = 0; i < M * K; i++) A[i] = (float)(i % 100) / 100.0f;
    for (int64_t i = 0; i < K * N; i++) B[i] = (float)(i % 50) / 50.0f;

    struct TestCase {
        const char* name;
        oil::math::Transpose trans;
    };
    TestCase cases[] = {
        {"AB",  oil::math::Transpose::None},
        {"AtB", oil::math::Transpose::TransA},
        {"ABt", oil::math::Transpose::TransB},
        {"AtBt",oil::math::Transpose::TransAB},
    };

    int iters = 1000;
    for (auto& tc : cases) {
        std::fill(C.begin(), C.end(), 0.0f);

        for (int i = 0; i < 5; i++)
            oil::math::gemm_tiled(1.0f, A.data(), K, B.data(), N,
                                  0.0f, C.data(), N, M, N, K, tc.trans);

        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            oil::math::gemm_tiled(1.0f, A.data(), K, B.data(), N,
                                  0.0f, C.data(), N, M, N, K, tc.trans);
        double dt = (now_sec() - t0) / iters;
        double flops = 2.0 * M * N * K;
        std::cout << "  " << std::setw(6) << tc.name
                  << "  time: " << std::fixed << std::setprecision(2) << (dt * 1e6) << " us"
                  << "  GFLOPS: " << std::fixed << std::setprecision(2) << (flops / dt * 1e-9)
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// bench_gemv: GEMV throughput
// ---------------------------------------------------------------------------
static void bench_gemv() {
    std::cout << "\n=== GEMV Benchmarks ===" << std::endl;

    std::vector<int64_t> sizes = {256, 512, 1024, 2048};

    for (int64_t n : sizes) {
        oil::Tensor A(oil::Shape{n, n}, oil::DType::F32);
        oil::Tensor x(oil::Shape{n}, oil::DType::F32);
        oil::Tensor y(oil::Shape{n}, oil::DType::F32);
        A.fill(1.0f);
        x.fill(1.0f);

        int warmup = 5;
        for (int i = 0; i < warmup; i++)
            oil::math::gemv(1.0f, A, x, 0.0f, y);

        int iters = 2000;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            oil::math::gemv(1.0f, A, x, 0.0f, y);
        double dt = (now_sec() - t0) / iters;

        double ops = 2.0 * n * n;
        std::cout << "  n=" << std::setw(5) << n
                  << "  time: " << std::fixed << std::setprecision(2) << (dt * 1e6) << " us"
                  << "  GFLOPS: " << std::fixed << std::setprecision(2) << (ops / dt * 1e-9) << std::endl;
    }
}

// ---------------------------------------------------------------------------
// bench_activations: ReLU, GELU, SiLU, sigmoid throughput
// ---------------------------------------------------------------------------
static void bench_activations() {
    std::cout << "\n=== Activation Function Benchmarks ===" << std::endl;

    std::vector<int64_t> sizes = {1024, 4096, 16384, 65536};

    struct ActCase {
        const char* name;
        void (*fn)(const oil::Tensor&, oil::Tensor&);
    };
    ActCase acts[] = {
        {"ReLU",  oil::math::relu},
        {"GELU",  oil::math::gelu},
        {"SiLU",  oil::math::silu},
        {"Sigmoid", oil::math::sigmoid},
    };

    for (int64_t n : sizes) {
        std::cout << "  n=" << n << std::endl;
        oil::Tensor x(oil::Shape{n}, oil::DType::F32);
        oil::Tensor y(oil::Shape{n}, oil::DType::F32);
        x.fill(1.0f);

        for (auto& a : acts) {
            int warmup = 5;
            for (int i = 0; i < warmup; i++) a.fn(x, y);

            int iters = 5000;
            double t0 = now_sec();
            for (int i = 0; i < iters; i++) a.fn(x, y);
            double dt = (now_sec() - t0) / iters;

            double ops = static_cast<double>(n);
            std::cout << "    " << std::setw(10) << a.name
                      << "  time: " << std::fixed << std::setprecision(2) << (dt * 1e6) << " us"
                      << "  GFLOPS: " << std::fixed << std::setprecision(2) << (ops / dt * 1e-9)
                      << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// bench_softmax: softmax throughput at various dimensions
// ---------------------------------------------------------------------------
static void bench_softmax() {
    std::cout << "\n=== Softmax Benchmarks ===" << std::endl;

    std::vector<int64_t> dims = {64, 128, 256, 512, 1024, 4096};
    const int rows = 100;

    for (int64_t dim : dims) {
        oil::Tensor x(oil::Shape{rows, dim}, oil::DType::F32);
        oil::Tensor y(oil::Shape{rows, dim}, oil::DType::F32);
        x.fill(1.0f);

        int warmup = 5;
        for (int i = 0; i < warmup; i++)
            oil::math::softmax(x, y, 1);

        int iters = 1000;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            oil::math::softmax(x, y, 1);
        double dt = (now_sec() - t0) / iters;

        std::cout << "  dim=" << std::setw(5) << dim
                  << "  time: " << std::fixed << std::setprecision(2) << (dt * 1e6) << " us"
                  << "  ops/s: " << std::scientific << std::setprecision(2)
                  << ((double)rows / dt) << std::endl;
    }
}

// ---------------------------------------------------------------------------
// bench_layer_norm: layer_norm throughput
// ---------------------------------------------------------------------------
static void bench_layer_norm() {
    std::cout << "\n=== LayerNorm Benchmarks ===" << std::endl;

    std::vector<int64_t> dims = {256, 512, 768, 1024, 2048};

    for (int64_t dim : dims) {
        oil::Tensor x(oil::Shape{32, dim}, oil::DType::F32);
        oil::Tensor gamma(oil::Shape{dim}, oil::DType::F32);
        oil::Tensor beta(oil::Shape{dim}, oil::DType::F32);
        oil::Tensor y(oil::Shape{32, dim}, oil::DType::F32);
        x.fill(0.5f);
        gamma.fill(1.0f);
        beta.fill(0.0f);

        int warmup = 5;
        for (int i = 0; i < warmup; i++)
            oil::math::layer_norm(x, gamma, beta, 1e-5f, y);

        int iters = 2000;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            oil::math::layer_norm(x, gamma, beta, 1e-5f, y);
        double dt = (now_sec() - t0) / iters;

        double ops = 32.0 * dim * 5.0;
        std::cout << "  dim=" << std::setw(5) << dim
                  << "  time: " << std::fixed << std::setprecision(2) << (dt * 1e6) << " us"
                  << "  GFLOPS: " << std::fixed << std::setprecision(2) << (ops / dt * 1e-9)
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// bench_fused_gemm_act: GEMM+activation fusion benefit
// ---------------------------------------------------------------------------
static void bench_fused_gemm_act() {
    std::cout << "\n=== Fused GEMM + Activation (256x256x256) ===" << std::endl;

    const int M = 256, N = 256, K = 256;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    auto unfused_fn = [&](oil::math::Activation act) {
        for (int i = 0; i < 5; i++)
            oil::math::gemm_tiled(1.0f, A.data(), K, B.data(), N,
                                  0.0f, C.data(), N, M, N, K);
        oil::Tensor Ct(oil::Shape{M, N}, oil::DType::F32);
        oil::Tensor Dt(oil::Shape{M, N}, oil::DType::F32);
        std::memcpy(Ct.data<float>(), C.data(), M * N * sizeof(float));

        int iters = 500;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            std::memcpy(Ct.data<float>(), C.data(), M * N * sizeof(float));
            switch (act) {
                case oil::math::Activation::ReLU: oil::math::relu(Ct, Dt); break;
                case oil::math::Activation::GELU: oil::math::gelu(Ct, Dt); break;
                case oil::math::Activation::SiLU: oil::math::silu(Ct, Dt); break;
                default: break;
            }
        }
        return (now_sec() - t0) / iters;
    };

    auto fused_fn = [&](oil::math::Activation act) {
        int iters = 500;
        for (int i = 0; i < 5; i++)
            oil::math::gemm_tiled_act(1.0f, A.data(), K, B.data(), N,
                                      0.0f, C.data(), N, M, N, K, act);

        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            oil::math::gemm_tiled_act(1.0f, A.data(), K, B.data(), N,
                                      0.0f, C.data(), N, M, N, K, act);
        return (now_sec() - t0) / iters;
    };

    struct { const char* name; oil::math::Activation act; } cases[] = {
        {"ReLU", oil::math::Activation::ReLU},
        {"GELU", oil::math::Activation::GELU},
        {"SiLU", oil::math::Activation::SiLU},
    };

    for (auto& c : cases) {
        double dt_unfused = unfused_fn(c.act);
        double dt_fused = fused_fn(c.act);
        std::cout << "  " << std::setw(8) << c.name
                  << "  unfused: " << std::fixed << std::setprecision(2) << (dt_unfused * 1e6) << " us"
                  << "  fused: " << std::fixed << std::setprecision(2) << (dt_fused * 1e6) << " us"
                  << "  speedup: " << std::fixed << std::setprecision(2) << (dt_unfused / dt_fused) << "x"
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== OIL Kernel Throughput Benchmarks ===" << std::endl;
    std::cout << "Minimum 1000 iterations per test. Timing via std::chrono." << std::endl;

    bench_gemm_comparison();
    bench_gemm_transpose();
    bench_dot_product();
    bench_gemv();
    bench_activations();
    bench_softmax();
    bench_layer_norm();
    bench_fused_gemm_act();

    std::cout << "\nDone." << std::endl;
    return 0;
}
