#include "oil/native_oil_moe.h"
#include "oil/random.h"
#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include "oil/test.h"

using namespace oil;

int main() {
    printf("=== NativeOIL MoE Tests ===\n\n");

    printf("--- Test 1: Model creation ---\n");
    native::NativeMoEConfig cfg;
    cfg.hidden_size = 128;
    cfg.num_heads = 4;
    cfg.ffn_hidden = 256;
    cfg.num_layers = 2;
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.vocab_size = 64;
    cfg.max_seq_len = 16;
    cfg.block_size = 32;

    native::NativeOILMoEModel model(cfg);
    printf("[DBG:TEST] after ctor, total_params=%llu\n", (unsigned long long)model.total_params()); fflush(stdout);
    TEST_CHECK(model.total_params() > 0, "model has params");
    printf("[DBG:TEST] TEST_CHECK passed\n"); fflush(stdout);

    printf("--- Test 2: Forward pass ---\n");
    RNG rng(12345);
    size_t B = 2, S = 8;
    std::vector<float> ids(B * S), tgts(B * S);
    for (size_t i = 0; i < B * S; i++) {
        ids[i] = (float)(int)(rng.uniform() * (cfg.vocab_size - 1));
        tgts[i] = (float)(int)((int)(ids[i] + 1) % cfg.vocab_size);
    }

    Tensor inp(Shape{(int64_t)B, (int64_t)S});
    std::memcpy(inp.data<float>(), ids.data(), B * S * sizeof(float));
    Tensor tgt(Shape{(int64_t)B, (int64_t)S});
    std::memcpy(tgt.data<float>(), tgts.data(), B * S * sizeof(float));

    AutogradEngine::set_enabled(true);
    Tensor logits = model.forward(inp);
    TEST_CHECK(logits.dim(2) == cfg.vocab_size, "logits shape correct");
    Tensor loss_t = AutogradEngine::cross_entropy_op(logits, tgt);
    float loss_val = *(const float*)loss_t.data();
    TEST_CHECK(loss_val > 0.0f, "loss is positive");
    TEST_CHECK(std::isfinite(loss_val), "loss is finite");
    printf("  Loss: %.4f\n", loss_val);

    printf("--- Test 3: Backward pass ---\n");
    model.backward(loss_t);
    auto& eng = AutogradEngine::instance();
    eng.clear();
    AutogradEngine::set_enabled(false);
    TEST_CHECK(true, "backward completed without crash");

    printf("--- Test 4: Training loop (20 steps) ---\n");
    float init_loss = 0.0f, final_loss = 0.0f;
    for (int step = 0; step < 20; step++) {
        for (size_t i = 0; i < B * S; i++) {
            ids[i] = (float)(int)(rng.uniform() * (cfg.vocab_size - 1));
            tgts[i] = (float)(int)((int)(ids[i] + 1) % cfg.vocab_size);
        }
        auto m = model.step(ids.data(), tgts.data(), B, S, 0.001f, 0.01f);
        if (step == 0) init_loss = m.loss;
        if (step == 19) final_loss = m.loss;
        if ((step + 1) % 5 == 0)
            printf("  Step %d: loss=%.4f params=%zu\n", step + 1, m.loss, m.total_params);
    }
    TEST_CHECK(final_loss < 100.0f, "loss did not explode");
    TEST_CHECK(std::isfinite(final_loss), "final loss is finite");

    printf("--- Test 5: Memory report ---\n");
    model.print_memory_report();

    return TEST_REPORT() > 0 ? 1 : 0;
}
