#include "oil/model.h"
#include "oil/autograd.h"
#include "oil/random.h"
#include "oil/tensor.h"
#include "oil/native_weight.h"
#include "oil/native_trainer.h"
#include "oil/test.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

using namespace oil;

int main() {
    TEST_SUITE("Native OIL Tests");
    printf("=== Native OIL Tests ===\n\n");

    // Test 1: Weight store basic operations
    printf("--- Test 1: Weight store ---\n");
    const size_t N = 256;
    native::NativeOILWeightStore store(N, 128);
    TEST_CHECK(store.size() == N, "size=256");
    TEST_CHECK(store.block_size() == 128, "block_size=128");

    std::vector<float> src(N), dst(N);
    for (size_t i = 0; i < N; i++) src[i] = (float)((int)i % 3 - 1);
    store.convert_from_fp32(src.data());
    store.dequantize(dst.data());
    TEST_CHECK(std::abs(dst[0] + store.get_scale(0)) < 1e-5f, "deq(-1) ≈ -scale");
    TEST_CHECK(std::abs(dst[1]) < 1e-5f, "deq(0) ≈ 0");

    // Test 2: Two-timescale update (Theorem 5d.3)
    printf("\n--- Test 2: Two-timescale update ---\n");
    native::NativeOILWeightStore store2(64, 64);
    std::vector<float> src2(64, 0.5f);
    store2.convert_from_fp32(src2.data());
    float s0 = store2.get_scale(0);
    uint8_t i0 = store2.get_index(0);
    printf("  Initial: idx=%d scale=%f\n", i0, s0);

    std::vector<float> small_grad(64, 0.01f);
    store2.apply_oil_update(small_grad.data(), 0.01f, 0.01f);
    TEST_CHECK(store2.get_scale(0) != s0, "scale changed");
    TEST_CHECK(store2.get_index(0) == i0, "index unchanged (dead zone)");

    // Test 3: CID allocation (OIL8 / Spark / OIL4)
    printf("\n--- Test 3: CID allocation ---\n");
    const size_t N3 = 2560;
    native::NativeOILWeightStore store_cid(N3, 128);
    std::vector<float> sens(N3);
    for (size_t i = 0; i < N3; i++) {
        if (i < 128) sens[i] = 1000.0f - (float)i;
        else if (i < N3 - 128) sens[i] = 500.0f - (float)(i - 128) * 0.1f;
        else sens[i] = 0.001f - (float)i * 1e-9f;
    }
    store_cid.reallocate_by_sensitivity(sens.data(), 0.05f, 0.90f);
    TEST_CHECK(store_cid.get_format(0) == native::NativeFormat::OIL8, "top block → OIL8");
    TEST_CHECK(store_cid.get_format(1280) == native::NativeFormat::OIL1, "middle block → OIL1");
    TEST_CHECK(store_cid.get_format(2559) == native::NativeFormat::OIL4, "bottom block → OIL4");

    // Test 4: Model forward/backward with autograd
    printf("\n--- Test 4: Model forward/backward ---\n");
    TransformerConfig cfg;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.ffn_hidden_size = 32;
    cfg.vocab_size = 16;
    cfg.max_seq_len = 8;

    DenseModel model(cfg);
    printf("  Params: %lld\n", (long long)model.param_count());
    TEST_CHECK(model.param_count() > 0, "model created");

    auto& engine = AutogradEngine::instance();
    std::vector<Tensor*> params;
    params.push_back(&model.tok_embeddings->weight);
    for (auto& layer : model.layers) {
        params.push_back(&layer->attention_norm.weight);
        params.push_back(&layer->attention.q_proj.weight);
        if (layer->attention.q_proj.bias.numel() > 0) params.push_back(&layer->attention.q_proj.bias);
        params.push_back(&layer->attention.k_proj.weight);
        if (layer->attention.k_proj.bias.numel() > 0) params.push_back(&layer->attention.k_proj.bias);
        params.push_back(&layer->attention.v_proj.weight);
        if (layer->attention.v_proj.bias.numel() > 0) params.push_back(&layer->attention.v_proj.bias);
        params.push_back(&layer->attention.o_proj.weight);
        if (layer->attention.o_proj.bias.numel() > 0) params.push_back(&layer->attention.o_proj.bias);
        params.push_back(&layer->ffn_norm.weight);
        params.push_back(&layer->ffn.gate_proj.weight);
        if (layer->ffn.gate_proj.bias.numel() > 0) params.push_back(&layer->ffn.gate_proj.bias);
        params.push_back(&layer->ffn.up_proj.weight);
        if (layer->ffn.up_proj.bias.numel() > 0) params.push_back(&layer->ffn.up_proj.bias);
        params.push_back(&layer->ffn.down_proj.weight);
        if (layer->ffn.down_proj.bias.numel() > 0) params.push_back(&layer->ffn.down_proj.bias);
    }
    params.push_back(&model.norm->weight);
    params.push_back(&model.lm_head->weight);
    if (model.lm_head->bias.numel() > 0) params.push_back(&model.lm_head->bias);
    for (auto* p : params) { p->requires_grad(true); engine.register_parameter(p); }

    RNG rng(12345);
    Tensor inp(Shape{1, (int64_t)cfg.max_seq_len});
    Tensor pos(Shape{1, (int64_t)cfg.max_seq_len});
    Tensor tgt(Shape{1, (int64_t)cfg.max_seq_len});
    for (int64_t s = 0; s < cfg.max_seq_len; s++) {
        float tok = (float)((int)(rng.uniform() * (cfg.vocab_size - 1)));
        inp.data<float>()[s] = tok;
        pos.data<float>()[s] = (float)s;
        tgt.data<float>()[s] = (float)((int)(tok + 1) % cfg.vocab_size);
    }
    engine.set_enabled(true);
    Tensor logits = model.forward(inp, pos);
    Tensor loss = AutogradEngine::cross_entropy_op(logits, tgt);
    engine.backward(loss);
    engine.clear();
    engine.set_enabled(false);

    bool has_grad = false;
    for (auto* p : params) { if (p->has_grad()) { has_grad = true; break; } }
    TEST_CHECK(has_grad, "gradients computed");
    printf("  Loss: %f\n", *(const float*)loss.data());

    // Test 5: NativeOILTrainer end-to-end
    printf("\n--- Test 5: NativeOILTrainer ---\n");
    native::NativeTrainConfig ncfg;
    ncfg.block_size = 64;
    ncfg.warmup_steps = 3;
    ncfg.max_steps = 5;
    ncfg.lr_scale = 1e-3f;
    ncfg.lr_weight = 1e-3f;
    ncfg.log_interval = 2;

    std::vector<std::vector<float>> train_data;
    for (size_t i = 0; i < 4; i++) {
        std::vector<float> seq((size_t)cfg.max_seq_len);
        for (size_t s = 0; s < (size_t)cfg.max_seq_len; s++)
            seq[s] = (float)((int)(rng.uniform() * (cfg.vocab_size - 1)));
        train_data.push_back(seq);
    }

    native::NativeOILTrainer trainer(&model, ncfg);
    TEST_CHECK(trainer.weight_store().size() > 0, "weight store initialized");

    trainer.warmup_phase(train_data);

    double initial_loss = 0.0, final_loss = 0.0;
    for (size_t step = 0; step < ncfg.max_steps; step++) {
        size_t idx = step % train_data.size();
        auto& seq = train_data[idx];
        auto m = trainer.train_step(seq.data(), seq.data(), 1, seq.size());
        if (step == 0) initial_loss = m.loss;
        if (step == ncfg.max_steps - 1) final_loss = m.loss;
        printf("  Step %zu: loss=%.4f frozen=%.0f%%\n", step+1, m.loss, m.frozen_fraction*100.0f);
    }
    TEST_CHECK(final_loss < 100.0f, "loss finite (training runs)");
    printf("  Loss: %.4f → %.4f\n", initial_loss, final_loss);

    return TEST_REPORT() > 0 ? 1 : 0;
}
