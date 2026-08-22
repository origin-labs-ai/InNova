// test_native_quant_moe.cpp — NativeQUANTMoEModel forward + training step
#include "quant/native_quant_moe.h"
#include "quant/tensor.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>

using namespace quant;
using namespace quant::native;

int main() {
    TEST_SUITE("native_quant_moe");
    printf("=== NativeQUANT MoE model test ===\n\n");

    NativeMoEConfig ncfg;
    ncfg.hidden_size = 32;
    ncfg.num_heads = 2;
    ncfg.ffn_hidden = 64;
    ncfg.num_layers = 1;
    ncfg.num_experts = 4;
    ncfg.top_k = 2;
    ncfg.block_size = 32;
    ncfg.vocab_size = 64;
    ncfg.max_seq_len = 32;

    NativeQUANTMoEModel model(ncfg);
    TEST_CHECK(model.total_params() > 0, "model has parameters");
    model.init_from_fp32();

    // Forward on random ids
    const int64_t B = 1, S = 16;
    Tensor ids(Shape{B, S}, DType::F32);
    unsigned int seed = 11;
    for (int64_t i = 0; i < B * S; i++) {
        seed = seed * 1103515245u + 12345u;
        ids.data<float>()[i] = (float)((seed >> 16) % (size_t)ncfg.vocab_size);
    }
    Tensor logits = model.forward(ids);
    TEST_CHECK(logits.numel() == B * S * ncfg.vocab_size, "logits shape B*S*V");
    bool finite = true;
    for (int64_t i = 0; i < logits.numel(); i++)
        if (!std::isfinite(logits.data<float>()[i])) { finite = false; break; }
    TEST_CHECK(finite, "logits finite");

    // One training step with learnable periodic targets
    std::vector<float> inp(B * S), tgt(B * S);
    for (int64_t s = 0; s < S; s++) {
        inp[(size_t)s] = (float)((s + 1) % (size_t)ncfg.vocab_size);
        tgt[(size_t)s] = (float)((s + 2) % (size_t)ncfg.vocab_size);
    }
    auto m0 = model.step(inp.data(), tgt.data(), (size_t)B, (size_t)S, 0.2f, 1.0f);
    auto m1 = model.step(inp.data(), tgt.data(), (size_t)B, (size_t)S, 0.2f, 1.0f);
    printf("  step loss: %.4f -> %.4f (grad_norm %.4f)\n", m0.loss, m1.loss, m1.grad_norm);
    TEST_CHECK(std::isfinite(m1.loss), "training loss finite");
    TEST_CHECK(m1.loss <= m0.loss + 1e-3f, "training loss does not increase");

    model.print_memory_report();

    int failures = TEST_REPORT();
    printf("\nNATIVE QUANT MOE TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
