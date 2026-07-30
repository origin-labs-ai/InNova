#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/gpu_compute.h"
#include "oil/test.h"
#include "oil/backend.h"

using namespace oil;
using namespace oil::gpu;
using namespace oil::backend;

bool has_dx = false;

static void test_gpu_init() {
    TEST_SUITE("GPU INIT");

    has_dx = is_directx_available();
    printf("  DirectX available at compile time: %s\n", has_dx ? "yes" : "no");

    bool gpu_ok = gpu::gpu_available();
    printf("  GPU compute initialized: %s\n", gpu_ok ? "yes" : "no");

    if (!gpu_ok) {
        gpu::init_gpu(GPUType::DIRECTX12, 0);
        gpu_ok = gpu::gpu_available();
        printf("  After init_gpu: %s\n", gpu_ok ? "yes" : "no");
    }

    if (gpu_ok) {
        TEST_CHECK(gpu_ok, "GPU compute initialized successfully");
        auto& dx = gpu::get_dx_compute();
        TEST_CHECK(dx.is_initialized(), "DirectXCompute::is_initialized()");
        int64_t mem_free = dx.memory_free();
        int64_t mem_total = dx.memory_total();
        printf("  GPU memory: free=%lld MB total=%lld MB\n",
               (long long)(mem_free / (1024*1024)),
               (long long)(mem_total / (1024*1024)));
        TEST_CHECK(mem_total > 0, "GPU memory_total > 0");
        TEST_CHECK(mem_free > 0, "GPU memory_free > 0");
    } else {
        printf("  SKIP: GPU initialization not available\n");
    }
}

static void test_gpu_memory() {
    TEST_SUITE("GPU MEMORY");
    if (!gpu::gpu_available()) { printf("  SKIP (no GPU)\n"); return; }

    auto& dx = gpu::get_dx_compute();

    // Allocate
    const int64_t N = 256;
    void* gpu_buf = dx.allocate(N * sizeof(float));
    TEST_CHECK(gpu_buf != nullptr, "gpu allocate");

    // Upload
    Tensor src({N}, DType::F32);
    for (int64_t i = 0; i < N; i++) src.data<float>()[i] = (float)i;
    dx.upload(src, gpu_buf);

    // Download
    Tensor dst({N}, DType::F32);
    dx.download(gpu_buf, dst);

    // Verify
    bool ok = true;
    for (int64_t i = 0; i < N && ok; i++)
        if (std::fabs(dst.data<float>()[i] - (float)i) > 1e-4f) ok = false;
    TEST_CHECK(ok, "gpu upload/download roundtrip");

    // Copy
    void* gpu_buf2 = dx.allocate(N * sizeof(float));
    dx.copy(gpu_buf2, gpu_buf, N * sizeof(float));
    Tensor dst2({N}, DType::F32);
    dx.download(gpu_buf2, dst2);
    ok = true;
    for (int64_t i = 0; i < N && ok; i++)
        if (std::fabs(dst2.data<float>()[i] - (float)i) > 1e-4f) ok = false;
    TEST_CHECK(ok, "gpu copy");

    dx.free(gpu_buf);
    dx.free(gpu_buf2);
}

static void test_gpu_relu() {
    TEST_SUITE("GPU RELU");
    if (!gpu::gpu_available()) { printf("  SKIP (no GPU)\n"); return; }

    auto& dx = gpu::get_dx_compute();
    const int64_t N = 128;

    Tensor x({N}, DType::F32);
    for (int64_t i = 0; i < N; i++) x.data<float>()[i] = (float)(i - 64);

    void* gpu_x = dx.allocate(N * sizeof(float));
    void* gpu_y = dx.allocate(N * sizeof(float));
    dx.upload(x, gpu_x);

    dx.relu(gpu_x, gpu_y, N);

    Tensor y({N}, DType::F32);
    dx.download(gpu_y, y);

    bool ok = true;
    for (int64_t i = 0; i < N && ok; i++) {
        float expected = (x.data<float>()[i] > 0) ? x.data<float>()[i] : 0.0f;
        if (std::fabs(y.data<float>()[i] - expected) > 1e-4f) ok = false;
    }
    TEST_CHECK(ok, "gpu relu correct");

    dx.free(gpu_x);
    dx.free(gpu_y);
}

static void test_gpu_elementwise() {
    TEST_SUITE("GPU ELEMENT-WISE");
    if (!gpu::gpu_available()) { printf("  SKIP (no GPU)\n"); return; }

    auto& dx = gpu::get_dx_compute();
    const int64_t N = 256;

    Tensor a({N}, DType::F32);
    Tensor b({N}, DType::F32);
    for (int64_t i = 0; i < N; i++) {
        a.data<float>()[i] = (float)i;
        b.data<float>()[i] = (float)(N - i);
    }

    void* gpu_a = dx.allocate(N * sizeof(float));
    void* gpu_b = dx.allocate(N * sizeof(float));
    void* gpu_c = dx.allocate(N * sizeof(float));
    dx.upload(a, gpu_a);
    dx.upload(b, gpu_b);

    // Test add
    dx.add(gpu_a, gpu_b, gpu_c, N);
    Tensor c_add({N}, DType::F32);
    dx.download(gpu_c, c_add);
    bool ok = true;
    for (int64_t i = 0; i < N && ok; i++)
        if (std::fabs(c_add.data<float>()[i] - (float)(i + (N - i))) > 1e-3f) ok = false;
    TEST_CHECK(ok, "gpu add correct");

    // Test mul
    dx.mul(gpu_a, gpu_b, gpu_c, N);
    Tensor c_mul({N}, DType::F32);
    dx.download(gpu_c, c_mul);
    ok = true;
    for (int64_t i = 0; i < N && ok; i++)
        if (std::fabs(c_mul.data<float>()[i] - (float)(i * (N - i))) > 1e-2f) ok = false;
    TEST_CHECK(ok, "gpu mul correct");

    // Test scale
    dx.scale(2.5f, gpu_a, gpu_c, N);
    Tensor c_scale({N}, DType::F32);
    dx.download(gpu_c, c_scale);
    ok = true;
    for (int64_t i = 0; i < N && ok; i++)
        if (std::fabs(c_scale.data<float>()[i] - 2.5f * i) > 1e-3f) ok = false;
    TEST_CHECK(ok, "gpu scale correct");

    dx.free(gpu_a);
    dx.free(gpu_b);
    dx.free(gpu_c);
}

static void test_gpu_softmax() {
    TEST_SUITE("GPU SOFTMAX");
    if (!gpu::gpu_available()) { printf("  SKIP (no GPU)\n"); return; }

    auto& dx = gpu::get_dx_compute();
    const int64_t ROWS = 4;
    const int64_t COLS = 8;
    const int64_t N = ROWS * COLS;

    Tensor x({ROWS, COLS}, DType::F32);
    for (int64_t i = 0; i < N; i++)
        x.data<float>()[i] = (float)((i * 7) % 10);  // varied values

    void* gpu_x = dx.allocate(N * sizeof(float));
    void* gpu_y = dx.allocate(N * sizeof(float));
    dx.upload(x, gpu_x);

    dx.softmax(gpu_x, gpu_y, ROWS, COLS);

    Tensor y({ROWS, COLS}, DType::F32);
    dx.download(gpu_y, y);

    bool ok = true;
    for (int64_t r = 0; r < ROWS && ok; r++) {
        float sum = 0;
        for (int64_t c = 0; c < COLS; c++)
            sum += y.data<float>()[r * COLS + c];
        if (std::fabs(sum - 1.0f) > 1e-3f) ok = false;
    }
    TEST_CHECK(ok, "gpu softmax rows sum to 1");

    dx.free(gpu_x);
    dx.free(gpu_y);
}

static void test_gpu_rms_norm() {
    TEST_SUITE("GPU RMS NORM");
    if (!gpu::gpu_available()) { printf("  SKIP (no GPU)\n"); return; }

    auto& dx = gpu::get_dx_compute();
    const int64_t N = 4;  // batch
    const int64_t D = 8;  // features

    Tensor x({N, D}, DType::F32);
    Tensor gamma({D}, DType::F32);
    for (int64_t i = 0; i < N * D; i++) x.data<float>()[i] = (float)((i * 3 + 1) % 7);
    for (int64_t i = 0; i < D; i++) gamma.data<float>()[i] = 1.0f;

    void* gpu_x = dx.allocate(N * D * sizeof(float));
    void* gpu_g = dx.allocate(D * sizeof(float));
    void* gpu_y = dx.allocate(N * D * sizeof(float));
    dx.upload(x, gpu_x);
    dx.upload(gamma, gpu_g);

    dx.rms_norm(gpu_x, gpu_g, gpu_y, 1e-5f, N, D);

    Tensor y({N, D}, DType::F32);
    dx.download(gpu_y, y);

    // Verify RMS norm: mean of y^2 should be ~1 per row
    bool ok = true;
    for (int64_t n = 0; n < N && ok; n++) {
        double ss = 0;
        for (int64_t d = 0; d < D; d++)
            ss += (double)y.data<float>()[n * D + d] * y.data<float>()[n * D + d];
        double rms = std::sqrt(ss / D);
        if (std::fabs(rms - 1.0) > 1e-3) ok = false;
    }
    TEST_CHECK(ok, "gpu rms_norm rows have RMS ~1");

    dx.free(gpu_x);
    dx.free(gpu_g);
    dx.free(gpu_y);
}

static void test_gpu_gemm() {
    TEST_SUITE("GPU GEMM");
    if (!gpu::gpu_available()) { printf("  SKIP (no GPU)\n"); return; }

    auto& dx = gpu::get_dx_compute();
    const int64_t M = 8, N = 12, K = 16;

    // Both CPU gemm and GPU HLSL shader access B in {K, N} layout:
    //   CPU: B[k * N + j]
    //   GPU: B[i * N + tid.x] where i=k, tid.x=j → same {K, N} row-major layout
    Tensor A({M, K}, DType::F32);
    Tensor B({K, N}, DType::F32);
    Tensor C({M, N}, DType::F32);

    for (int64_t i = 0; i < M * K; i++) A.data<float>()[i] = ((float)(i % 5)) / 5.0f;
    for (int64_t i = 0; i < K * N; i++) B.data<float>()[i] = ((float)((i * 3) % 7)) / 7.0f;
    C.fill(0.0f);

    // Reference: CPU matmul (C = A * B)
    Tensor C_ref({M, N}, DType::F32);
    math::gemm(1.0f, A, B, 0.0f, C_ref);

    // GPU GEMM
    void* gpu_A = dx.allocate(M * K * sizeof(float));
    void* gpu_B = dx.allocate(K * N * sizeof(float));  // {K, N} layout — same as CPU
    void* gpu_C = dx.allocate(M * N * sizeof(float));

    dx.upload(A, gpu_A);
    dx.upload(B, gpu_B);
    dx.upload(C, gpu_C);

    dx.gemm(1.0f, gpu_A, gpu_B, 0.0f, gpu_C, M, N, K);

    Tensor C_gpu({M, N}, DType::F32);
    dx.download(gpu_C, C_gpu);

    bool ok = true;
    for (int64_t i = 0; i < M * N && ok; i++) {
        if (std::fabs(C_gpu.data<float>()[i] - C_ref.data<float>()[i]) > 1e-3f) {
            ok = false;
            printf("  Mismatch at [%lld]: GPU=%f CPU=%f\n",
                   (long long)i, C_gpu.data<float>()[i], C_ref.data<float>()[i]);
        }
    }
    TEST_CHECK(ok, "gpu gemm matches cpu gemm");

    dx.free(gpu_A);
    dx.free(gpu_B);
    dx.free(gpu_C);
}

static void test_gpu_shutdown() {
    TEST_SUITE("GPU SHUTDOWN");
    if (!gpu::gpu_available()) {
        printf("  SKIP (no GPU)\n");
        return;
    }
    TEST_CHECK(gpu::gpu_available(), "gpu still available");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("InNova — GPU Compute Test Suite\n");
    printf("====================================\n");

    has_dx = is_directx_available();
    printf("DirectX compile-time support: %s\n", has_dx ? "yes" : "no");

    test_gpu_init();
    test_gpu_memory();
    test_gpu_relu();
    test_gpu_elementwise();
    test_gpu_softmax();
    test_gpu_rms_norm();
    test_gpu_gemm();
    test_gpu_shutdown();

    printf("\n====================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
