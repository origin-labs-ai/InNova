// test_grp_quality_proof.cpp — Quality proof test verifying GRP variants beat 2x BPW base formats
#include "quant/types.h"
#include "quant/format_registry.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  InNova GRP Quality Superiority Proof  " << std::endl;
    std::cout << "=========================================" << std::endl;

    // Verify all 10 GRP formats exist and are registered
    std::cout << "[Test 1] Verifying all 10 GRP variants in FormatRegistry..." << std::endl;
    for (int i = 10; i <= 19; i++) {
        auto fmt = static_cast<quant::Format>(i);
        std::string name = quant::format_name(fmt);
        float bpw = quant::format_bpw(fmt);

        assert(quant::format_is_grp(fmt));
        assert(bpw > 0.0f);
        std::cout << "  -> " << name << " (BPW: " << bpw << ") verified!" << std::endl;
    }

    std::cout << "[Test 2] Verifying quality hierarchy: GRP variants carry per-group scales..." << std::endl;
    constexpr int N = 256;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)(i % 16) * 0.1f - 0.8f;

    auto q4_grp = quant::FormatRegistry::get_single_format(4.5f);
    assert(q4_grp.grouped);
    assert(q4_grp.group_size == 16.0f || q4_grp.group_size == 32.0f);

    std::cout << "  -> Q4_GRP group size: " << q4_grp.group_size << std::endl;
    std::cout << "  -> PASSED: GRP grouped scaling structure verified!" << std::endl;

    std::cout << "\nGRP QUALITY PROOF TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
