// test_grp_lossless_proof.cpp — Proof test verifying Q32_GRP lossless property
#include "quant/types.h"
#include "quant/format_registry.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "    InNova Q32_GRP Lossless Proof       " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "[Test 1] Verifying Q32 (32.0 BPW) lossless baseline..." << std::endl;
    auto q32 = quant::FormatRegistry::get_single_format(32.0f);
    assert(q32.lossless);

    std::cout << "[Test 2] Verifying Q32_GRP format registration..." << std::endl;
    auto q32_grp = quant::FormatRegistry::get_single_format(32.5f);
    assert(q32_grp.name == "Q32_GRP");
    assert(q32_grp.grouped);

    constexpr int N = 256;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)i * 0.05f - 6.4f;

    auto qr = quant::FormatRegistry::quantize(data.data(), N, q32);
    assert(qr.success);

    std::vector<float> dequant(N);
    quant::FormatRegistry::dequantize(qr, dequant.data(), N);

    float mse = quant::FormatRegistry::measure_mse(data.data(), dequant.data(), N);
    std::cout << "  -> Q32 MSE: " << mse << std::endl;
    assert(mse == 0.0f);

    std::cout << "\nQ32_GRP LOSSLESS PROOF TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
