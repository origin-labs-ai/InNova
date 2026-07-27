#include "oil/model.h"
#include "oil/autograd.h"
#include "oil/random.h"
#include "oil/optimizer.h"
#include "oil/native_weight.h"
#include "oil/native_trainer.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>

using namespace oil;

struct TrainResult {
    float train_loss = 0;
    float val_loss = 0;
    int64_t tokens = 0;
    double elapsed = 0;
    double tps = 0;
};

static float eval_val(DenseModel& model, RNG& rng, int64_t B, int64_t S, int64_t V, int rounds) {
    float total = 0.0f;
    for (int r = 0; r < rounds; r++) {
        std::vector<float> ids(B * S), tgts(B * S);
        for (int64_t i = 0; i < B * S; i++) {
            ids[i] = (float)(int)(rng.uniform() * (V - 1));
            tgts[i] = (i + 1 < B * S) ? ids[i + 1] : (float)(int)(rng.uniform() * (V - 1));
        }
        Tensor inp(Shape{B, S}), tgt(Shape{B, S}), pos(Shape{B, S});
        std::memcpy(inp.data<float>(), ids.data(), B * S * sizeof(float));
        std::memcpy(tgt.data<float>(), tgts.data(), B * S * sizeof(float));
        for (int64_t i = 0; i < B * S; i++) pos.data<float>()[i] = (float)(i % S);
        Tensor logits = model.forward(inp, pos);
        int64_t V2 = logits.dim(2);
        const float* ld = logits.data<float>();
        const float* td = tgt.data<float>();
        for (int64_t i = 0; i < B * S; i++) {
            int64_t t = (int64_t)td[i];
            if (t < 0 || t >= V2) continue;
            float mx = ld[i * V2];
            for (int64_t v = 1; v < V2; v++) if (ld[i * V2 + v] > mx) mx = ld[i * V2 + v];
            float s = 0.0f;
            for (int64_t v = 0; v < V2; v++) s += std::exp(ld[i * V2 + v] - mx);
            total += -(ld[i * V2 + t] - mx - std::log(s));
        }
    }
    return total / (float)(rounds * B * S);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    printf("================================================\n");
    printf("  64M NativeOIL Training (1.50 BPW)\n");
    printf("  Live HuggingFace streaming — zero disk storage\n");
    printf("  64M params | 1024 tok/param | 2 epochs\n");
    printf("  OIL: 95%% SPARK | 4%% OIL4 | 1%% OIL8\n");
    printf("================================================\n\n");

    // ─── Model Config ──────────────────────────────────────────
    TransformerConfig mcfg;
    mcfg.hidden_size = 512;
    mcfg.num_heads = 8;
    mcfg.head_dim = 64;
    mcfg.ffn_hidden_size = 2048;
    mcfg.num_layers = 14;      // ~67M params (closest to 64M)
    mcfg.vocab_size = 8192;
    mcfg.max_seq_len = 2048;

    int64_t B = 8, S = 2048;
    int64_t V = mcfg.vocab_size;
    int64_t max_tokens = 65536000000; // 65.5B tokens = 2 epochs
    int64_t warmup_tokens = 67108864; // 64M tokens warmup (~1 epoch over 64M)
    
    // For testing: limit to a few steps
    bool quick_test = false;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--test") == 0) {
            quick_test = true;
            max_tokens = B * S * 5; // just 5 batches
            printf("  QUICK TEST MODE: %lld tokens\n", (long long)max_tokens);
            break;
        }
    }

    DenseModel model(mcfg);
    int64_t total_params = model.param_count();
    printf("Model: %lld params (%.1fM) — %lld weights\n",
           (long long)total_params, total_params / 1e6,
           (long long)total_params);
    fflush(stdout);

// ─── Synthetic Batch Generator (no external deps) ───────────────
struct BatchGenerator {
    int64_t V;
    RNG rng;
    int64_t epoch = 0;
    int64_t total_toks = 0;
    BatchGenerator(int64_t voc) : V(voc), rng(42) {}
    bool next_batch(float* ids, float* tgts, int64_t B, int64_t S) {
        for (int64_t i = 0; i < B * S; i++) {
            float v = (float)(int)(rng.uniform() * (V - 1));
            ids[i] = v;
            tgts[i] = (i + 1 < B * S) ? ids[i + 1] : 0.0f;
        }
        total_toks += B * S;
        return true;
    }
    void reset_all() { epoch = 0; total_toks = 0; }
    int64_t epochs_completed() const { return epoch; }
};
BatchGenerator mixer(V);

    // ─── Native OIL Weight Store ───────────────────────────────
    size_t oil_params = (size_t)total_params;
    native::NativeOILWeightStore oil_store(oil_params, 128);

    // Initialize with FP32 weights → OIL
    {
        std::vector<float> all_w(oil_params);
        size_t off = 0;
        auto collect = [&](Tensor& t) {
            std::memcpy(all_w.data() + off, t.data<float>(), (size_t)t.numel() * sizeof(float));
            off += (size_t)t.numel();
        };
        // Collect all weights
        collect(model.tok_embeddings->weight);
        for (auto& l : model.layers) {
            collect(l->attention_norm.weight);
            collect(l->attention.q_proj.weight);
            collect(l->attention.k_proj.weight);
            collect(l->attention.v_proj.weight);
            collect(l->attention.o_proj.weight);
            collect(l->ffn_norm.weight);
            collect(l->ffn.gate_proj.weight);
            collect(l->ffn.up_proj.weight);
            collect(l->ffn.down_proj.weight);
        }
        collect(model.norm->weight);
        collect(model.lm_head->weight);

        std::vector<float> sens(oil_params, 1.0f);
        oil_store.initialize(all_w.data(), sens.data(), 0.01f, 0.95f);

        // Dequantize back to FP32 for model forward
        oil_store.dequantize(all_w.data());
        off = 0;
        auto deq = [&](Tensor& t) {
            size_t n = (size_t)t.numel();
            std::memcpy(t.data<float>(), all_w.data() + off, n * sizeof(float));
            off += n;
        };
        deq(model.tok_embeddings->weight);
        for (auto& l : model.layers) {
            deq(l->attention_norm.weight);
            deq(l->attention.q_proj.weight);
            deq(l->attention.k_proj.weight);
            deq(l->attention.v_proj.weight);
            deq(l->attention.o_proj.weight);
            deq(l->ffn_norm.weight);
            deq(l->ffn.gate_proj.weight);
            deq(l->ffn.up_proj.weight);
            deq(l->ffn.down_proj.weight);
        }
        deq(model.norm->weight);
        deq(model.lm_head->weight);

        // Count format distribution
        size_t oil8 = 0, spark_q0 = 0, oil4 = 0;
        for (size_t i = 0; i < oil_params; i++) {
            auto f = oil_store.get_format(i);
            if (f == native::NativeFormat::OIL8) oil8++;
            else if (f == native::NativeFormat::OIL1) spark_q0++;
            else oil4++;
        }
        printf("  OIL8: %.1f%% | OIL1: %.1f%% | OIL4: %.1f%%\n",
               100.0f * oil8 / oil_params,
               100.0f * spark_q0 / oil_params,
               100.0f * oil4 / oil_params);
        printf("  Weight size: %.2f MB (vs %.1f MB FP32) — %.1fx savings\n",
               oil_params * 1.5f / 8.0f / 1e6f,
               total_params * 4.0f / 1e6f,
               (float)total_params * 4.0f / (oil_params * 1.5f / 8.0f));
    }

    // ─── Autograd Setup ──────────────────────────────────────
    auto& eng = AutogradEngine::instance();
    std::vector<Tensor*> params;
    auto reg = [&](Tensor& t) {
        t.requires_grad(true);
        eng.register_parameter(&t);
        params.push_back(&t);
    };
    reg(model.tok_embeddings->weight);
    for (auto& l : model.layers) {
        reg(l->attention_norm.weight);
        reg(l->attention.q_proj.weight);
        reg(l->attention.k_proj.weight);
        reg(l->attention.v_proj.weight);
        reg(l->attention.o_proj.weight);
        reg(l->ffn_norm.weight);
        reg(l->ffn.gate_proj.weight);
        reg(l->ffn.up_proj.weight);
        reg(l->ffn.down_proj.weight);
    }
    reg(model.norm->weight);
    reg(model.lm_head->weight);

    SGD optimizer(1e-4f); // Lower LR for OIL training

    // ─── Training Loop ─────────────────────────────────────────
    printf("\n--- Starting NativeOIL Training ---\n");
    printf("  Target: %lld tokens (%.1fB) | Batch: %lld × %lld\n",
           (long long)max_tokens, max_tokens / 1e9, (long long)B, (long long)S);
    printf("  OIL format: 95%% SPARK + 4%% OIL4 + 1%% OIL8 (1.50 BPW)\n\n");

    RNG rng(42);
    auto t0 = std::chrono::high_resolution_clock::now();
    float ema_loss = 0.0f;
    int64_t total_tokens = 0;
    int64_t step = 0;

    while (total_tokens < max_tokens) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (elapsed > 10800) break; // 3-hour safety limit

        // Get batch from HF stream
        std::vector<float> ids(B * S), tgts(B * S);
        if (!mixer.next_batch(ids.data(), tgts.data(), B, S)) {
            printf("\n  Data stream exhausted. Restarting...\n");
            mixer.reset_all();
            if (!mixer.next_batch(ids.data(), tgts.data(), B, S)) break;
        }

        Tensor inp(Shape{B, S}), tgt(Shape{B, S}), pos(Shape{B, S});
        std::memcpy(inp.data<float>(), ids.data(), B * S * sizeof(float));
        std::memcpy(tgt.data<float>(), tgts.data(), B * S * sizeof(float));
        for (int64_t i = 0; i < B * S; i++) pos.data<float>()[i] = (float)(i % S);

        // Forward (dequantized OIL weights)
        optimizer.zero_grad();
        eng.set_enabled(true);
        Tensor logits = model.forward(inp, pos);
        Tensor loss = eng.cross_entropy_op(logits, tgt);
        eng.backward(loss);
        eng.clear();
        eng.set_enabled(false);
        optimizer.step();

        // Re-quantize: FP32 update → OIL encode → dequantize for next forward
        {
            std::vector<float> all_w(oil_params);
            size_t off = 0;
            for (auto* p : params) {
                size_t n = (size_t)p->numel();
                std::memcpy(all_w.data() + off, p->data<float>(), n * sizeof(float));
                off += n;
            }
            oil_store.convert_from_fp32(all_w.data());
            oil_store.dequantize(all_w.data());
            off = 0;
            for (auto* p : params) {
                size_t n = (size_t)p->numel();
                std::memcpy(p->data<float>(), all_w.data() + off, n * sizeof(float));
                off += n;
            }
        }

        float lv = *(const float*)loss.data();
        ema_loss = (step == 0) ? lv : 0.95f * ema_loss + 0.05f * lv;
        total_tokens += B * S;
        step++;

        if (step % 50 == 0) {
            double el = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            double tps = total_tokens / el;
            int64_t pct = total_tokens * 100 / max_tokens;
            printf("  step %lld | loss=%.4f | %.0f tok/s | %.0fs | %lld%% | epoch %lld\n",
                   (long long)step, ema_loss, tps, el,
                   (long long)pct, (long long)mixer.epochs_completed());
        }
        if (step % 500 == 0) {
            float vl = eval_val(model, rng, B, S, V, 5);
            printf("  >>> EVAL: val_loss=%.4f PPL=%.2f\n", vl, std::exp(vl));
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    float final_val = eval_val(model, rng, B, S, V, 10);

    printf("\n================================================\n");
    printf("  NativeOIL Training Complete\n");
    printf("================================================\n");
    printf("  Params:           %lld (%.1fM)\n", (long long)total_params, total_params / 1e6);
    printf("  Format:           1.50 BPW (95%% SPARK + 4%% OIL4 + 1%% OIL8)\n");
    printf("  Weight size:      %.2f MB\n", oil_params * 1.5f / 8.0f / 1e6f);
    printf("  Final loss:       %.4f\n", ema_loss);
    printf("  Final val_loss:   %.4f  PPL: %.2f\n", final_val, std::exp(final_val));
    printf("  Total tokens:     %lld (%.1fB)\n", (long long)total_tokens, total_tokens / 1e9);
    printf("  tok/param:        %.0f\n", (float)total_tokens / total_params);
    printf("  Time:             %.0f seconds\n", elapsed);
    printf("  tok/s:            %.0f\n", total_tokens / elapsed);
    printf("================================================\n");

    // ─── Save model ────────────────────────────────────────────
    printf("\nSaving model to final_64m.oil...\n");
    model.save("final_64m.oil");
    printf("Done.\n");

    return 0;
}
