// test_math.cpp — Unit test for InNova math library (activations, norms, BLAS)
#include "quant/math.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "      InNova Math Library Unit Test      " << std::endl;
    std::cout << "=========================================" << std::endl;

    constexpr int N = 256;
    std::vector<float> input(N);
    std::vector<float> output(N);

    for (int i = 0; i < N; i++) input[i] = (float)i * 0.05f - 6.4f;

    std::cout << "[Test 1] Testing ReLU activation..." << std::endl;
    quant::math::vec_relu(output.data(), input.data(), N);
    for (int i = 0; i < N; i++) {
        float expected = std::max(0.0f, input[i]);
        assert(std::abs(output[i] - expected) < 1e-5f);
    }
    std::cout << "  -> PASSED: ReLU verified!" << std::endl;

    std::cout << "[Test 2] Testing SiLU activation..." << std::endl;
    quant::math::vec_silu(output.data(), input.data(), N);
    for (int i = 0; i < N; i++) {
        float sig = 1.0f / (1.0f + std::exp(-input[i]));
        float expected = input[i] * sig;
        assert(std::abs(output[i] - expected) < 1e-4f);
    }
    std::cout << "  -> PASSED: SiLU verified!" << std::endl;

    std::cout << "[Test 3] Testing RMSNorm..." << std::endl;
    std::vector<float> weight(N, 1.0f);
    quant::math::vec_rms_norm(output.data(), input.data(), weight.data(), N, 1e-5f);
    double sq_sum = 0.0;
    for (int i = 0; i < N; i++) sq_sum += output[i] * output[i];
    double rms = std::sqrt(sq_sum / N);
    std::cout << "  -> RMS of normalized output: " << rms << std::endl;
    assert(std::abs(rms - 1.0) < 1e-3);
    std::cout << "  -> PASSED: RMSNorm verified!" << std::endl;

    std::cout << "\nALL MATH TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
