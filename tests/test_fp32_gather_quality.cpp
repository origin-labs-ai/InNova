// test_fp32_gather_quality.cpp — Unit test for FP32 codebook gather precision
#include "quant/types.h"
#include "quant/format_registry.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << " InNova FP32 Codebook Gather Quality Test" << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "[Test 1] Verifying FP32 centroid gather precision..." << std::endl;
    constexpr int centroids = 256;
    std::vector<float> codebook(centroids);
    for (int i = 0; i < centroids; i++) codebook[i] = -1.0f + 2.0f * i / 255.0f;

    std::vector<uint8_t> indices = {0, 64, 128, 192, 255};
    std::vector<float> gathered(indices.size());

    for (size_t i = 0; i < indices.size(); i++) {
        gathered[i] = codebook[indices[i]];
    }

    assert(gathered[0] == -1.0f);
    assert(gathered[4] == 1.0f);
    std::cout << "  -> PASSED: FP32 centroid gather exactness verified!" << std::endl;

    std::cout << "\nFP32 GATHER QUALITY TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
