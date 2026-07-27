#include "oil/model.h"
#include "oil/autograd.h"
#include "oil/random.h"
#include "oil/optimizer.h"
#include "oil/native_weight.h"
#include <cstdio>
#include <chrono>
using namespace oil;
int main() {
    TransformerConfig mcfg;
    mcfg.hidden_size = 512; mcfg.num_heads = 8; mcfg.head_dim = 64;
    mcfg.ffn_hidden_size = 2048; mcfg.num_layers = 14;
    mcfg.vocab_size = 8192; mcfg.max_seq_len = 2048;
    int64_t B=8, S=2048, V=mcfg.vocab_size;
    DenseModel model(mcfg);
    int64_t total_params = model.param_count();
    printf("Model: %lld params\n", (long long)total_params);
    // Register params
    auto& eng = AutogradEngine::instance();
    auto reg = [&](Tensor& t) { t.requires_grad(true); eng.register_parameter(&t); };
    reg(model.tok_embeddings->weight);
    for (auto& l : model.layers) {
        reg(l->attention_norm.weight); reg(l->attention.q_proj.weight);
        reg(l->attention.k_proj.weight); reg(l->attention.v_proj.weight);
        reg(l->attention.o_proj.weight); reg(l->ffn_norm.weight);
        reg(l->ffn.gate_proj.weight); reg(l->ffn.up_proj.weight);
        reg(l->ffn.down_proj.weight);
    }
    reg(model.norm->weight); reg(model.lm_head->weight);
    SGD optimizer(1e-4f);
    RNG rng(42);
    int warmup = 5, measured = 20;
    for (int step = 0; step < warmup + measured; step++) {
        std::vector<float> ids(B*S), tgts(B*S);
        for (int64_t i = 0; i < B*S; i++) {
            ids[i] = (float)(int)(rng.uniform() * (V-1));
            tgts[i] = (i+1 < B*S) ? ids[i+1] : (float)(int)(rng.uniform()*(V-1));
        }
        Tensor inp(Shape{B,S}), tgt(Shape{B,S}), pos(Shape{B,S});
        std::memcpy(inp.data<float>(), ids.data(), B*S*sizeof(float));
        std::memcpy(tgt.data<float>(), tgts.data(), B*S*sizeof(float));
        for (int64_t i = 0; i < B*S; i++) pos.data<float>()[i] = (float)(i % S);
        if (step == warmup) {
            printf("\n--- Benchmarking %d steps ---\n", measured);
            fflush(stdout);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        optimizer.zero_grad();
        eng.set_enabled(true);
        Tensor logits = model.forward(inp, pos);
        Tensor loss = eng.cross_entropy_op(logits, tgt);
        eng.backward(loss);
        eng.clear();
        eng.set_enabled(false);
        optimizer.step();
        if (step >= warmup) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            double tps = (B*S) / (ms/1000.0);
            printf("  step %d: %.0f ms, %.0f tok/s\n", step, ms, tps);
            fflush(stdout);
        }
    }
    return 0;
}
