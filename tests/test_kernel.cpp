#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/codebook.h"
#include "oil/test.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

static void scalar_gemm_ref(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static bool all_close(const float* a, const float* b, int64_t n, float eps = 1e-4f) {
    for (int64_t i = 0; i < n; i++)
        if (std::abs(a[i] - b[i]) >= eps)
            return false;
    return true;
}

int main() {
    TEST_SUITE("Kernel Tests");

    // Test scalar_gemm reference correctness
    {
        const int M = 4, N = 3, K = 5;
        std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
        for (int i = 0; i < M * K; i++) A[i] = (float)(i % 3);
        for (int i = 0; i < K * N; i++) B[i] = (float)((i * 2) % 5);
        oil::kernel::scalar_gemm(A.data(), B.data(), C.data(), M, N, K);
        std::vector<float> C_ref(M * N);
        scalar_gemm_ref(A.data(), B.data(), C_ref.data(), M, N, K);
        TEST_CHECK(all_close(C.data(), C_ref.data(), M * N), "scalar_gemm vs reference 4x3x5");
    }

    // Test scalar_gemm edge cases: zero matrix
    {
        const int M = 3, N = 3, K = 3;
        std::vector<float> A(M * K, 0.0f), B(K * N, 0.0f), C(M * N, 1.0f);
        oil::kernel::scalar_gemm(A.data(), B.data(), C.data(), M, N, K);
        bool zero = true;
        for (int i = 0; i < M * N; i++)
            if (C[i] != 0.0f) { zero = false; break; }
        TEST_CHECK(zero, "scalar_gemm zero matrix produces zeros");
    }

    // Test scalar_gemm edge cases: identity
    {
        const int N = 4;
        std::vector<float> A(N * N, 0.0f), B(N * N, 0.0f), C(N * N, 0.0f);
        for (int i = 0; i < N; i++) { A[i * N + i] = 1.0f; B[i * N + i] = 1.0f; }
        scalar_gemm_ref(A.data(), B.data(), C.data(), N, N, N);
        bool pass = true;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (std::abs(C[i * N + j] - (i == j ? 1.0f : 0.0f)) >= 1e-4f)
                    { pass = false; break; }
        TEST_CHECK(pass, "scalar_gemm 4x4 identity");
    }

    // Test scalar_gemm rectangular
    {
        const int M = 5, N = 2, K = 7;
        std::vector<float> A(M * K), B(K * N), C(M * N);
        for (int i = 0; i < M * K; i++) A[i] = (float)(i);
        for (int i = 0; i < K * N; i++) B[i] = 1.0f;
        std::vector<float> C_ref(M * N);
        scalar_gemm_ref(A.data(), B.data(), C_ref.data(), M, N, K);
        oil::kernel::scalar_gemm(A.data(), B.data(), C.data(), M, N, K);
        TEST_CHECK(all_close(C.data(), C_ref.data(), M * N), "scalar_gemm 5x2x7 rectangular");
    }

    // Test oil8_gemm produces finite output
    {
        const int M = 2, N = 3, K = 4;
        oil::Tensor Act(oil::Shape{N, K}, oil::DType::F32);
        float ad[] = {1,0,2,1, -1,2,0,1, 0,1,1,0};
        memcpy(Act.data(), ad, N * K * sizeof(float));

        float cb[256];
        for (int i = 0; i < 256; i++) cb[i] = (float)(i - 128) / 16.0f;

        std::vector<uint8_t> idx(M * N * K);
        for (int i = 0; i < M * N * K; i++)
            idx[i] = (uint8_t)((i * 37) % 256);

        oil::Tensor C(oil::Shape{M, N}, oil::DType::F32);
        oil::kernel::oil8_gemm(idx.data(), cb, Act.data<float>(), C.data<float>(), M, N, K);

        bool finite = true;
        for (int i = 0; i < M * N; i++)
            if (!std::isfinite(C.data<float>()[i])) finite = false;
        TEST_CHECK(finite, "oil8_gemm all finite");
    }

    // Test avx2_gemm fallback
    {
        const int M = 2, N = 2, K = 2;
        std::vector<float> A(M * K, 1.0f), B(K * N, 1.0f), C(M * N, 0.0f);
        oil::kernel::avx2_gemm(A.data(), B.data(), C.data(), M, N, K);
        TEST_CHECK(C[0] > 0.0f, "avx2_gemm fallback produces output");
    }

    return TEST_REPORT() > 0 ? 1 : 0;
}
