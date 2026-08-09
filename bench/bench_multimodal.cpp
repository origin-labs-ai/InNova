#include "quant/multimodal.h"
#include "quant/model.h"
#include "quant/tensor.h"
#include "quant/transformer.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <random>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

static quant::Tensor make_random(int64_t B, int64_t S, int64_t D) {
    quant::Tensor t(quant::Shape{B, S, D}, quant::DType::F32);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    float* d = t.data<float>();
    for (int64_t i = 0; i < B * S * D; i++) d[i] = dist(rng);
    return t;
}

static void bench_vision_encoder() {
    std::cout << "\n=== Vision Encoder Throughput ===\n";
    int64_t D = 768;
    std::vector<int64_t> seq_lens = {64, 128, 256, 512};

    quant::TransformerConfig cfg;
    cfg.hidden_size = D;
    cfg.num_layers = 6;
    cfg.num_heads = 12;
    cfg.head_dim = 64;
    cfg.ffn_hidden_size = D * 4;
    cfg.vocab_size = 32000;
    cfg.max_seq_len = 1024;
    cfg.activation = quant::Activation::SiLU;
    quant::DenseModel model(cfg);

    quant::ModalityProjection proj(D);

    for (int64_t S : seq_lens) {
        auto feat = make_random(1, S, D);
        auto pfeat = proj.project_vision(feat);

        int warmup = 5;
        for (int i = 0; i < warmup; i++) {
            quant::Tensor pos({1, S}, quant::DType::F32);
            float* pd = pos.data<float>();
            for (int64_t j = 0; j < S; j++) pd[j] = (float)j;
            model.forward(pfeat, pos);
        }

        int iters = 20;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            quant::Tensor pos({1, S}, quant::DType::F32);
            float* pd = pos.data<float>();
            for (int64_t j = 0; j < S; j++) pd[j] = (float)j;
            model.forward(pfeat, pos);
        }
        double dt = (now_sec() - t0) / iters;
        double tok_s = S / dt;

        std::cout << "  S=" << std::setw(4) << S
                  << "  " << std::fixed << std::setprecision(0) << tok_s << " tok/s"
                  << "  " << std::fixed << std::setprecision(3) << dt * 1000 << " ms\n";
    }
}

static void bench_audio_encoder() {
    std::cout << "\n=== Audio Encoder Throughput ===\n";
    int64_t D = 512;
    std::vector<int64_t> seq_lens = {64, 128, 256, 512};

    quant::TransformerConfig cfg;
    cfg.hidden_size = D;
    cfg.num_layers = 4;
    cfg.num_heads = 8;
    cfg.head_dim = 64;
    cfg.ffn_hidden_size = D * 4;
    cfg.vocab_size = 32000;
    cfg.max_seq_len = 1024;
    cfg.activation = quant::Activation::SiLU;
    cfg.norm_eps = 1e-5f;
    quant::DenseModel model(cfg);

    quant::ModalityProjection proj(D);

    for (int64_t S : seq_lens) {
        auto feat = make_random(1, S, D);
        auto pfeat = proj.project_audio(feat);

        int warmup = 5;
        for (int i = 0; i < warmup; i++) {
            quant::Tensor pos({1, S}, quant::DType::F32);
            float* pd = pos.data<float>();
            for (int64_t j = 0; j < S; j++) pd[j] = (float)j;
            model.forward(pfeat, pos);
        }

        int iters = 20;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            quant::Tensor pos({1, S}, quant::DType::F32);
            float* pd = pos.data<float>();
            for (int64_t j = 0; j < S; j++) pd[j] = (float)j;
            model.forward(pfeat, pos);
        }
        double dt = (now_sec() - t0) / iters;
        double tok_s = S / dt;

        std::cout << "  S=" << std::setw(4) << S
                  << "  " << std::fixed << std::setprecision(0) << tok_s << " tok/s"
                  << "  " << std::fixed << std::setprecision(3) << dt * 1000 << " ms\n";
    }
}

static void bench_multimodal_fusion() {
    std::cout << "\n=== Multimodal Fusion Throughput ===\n";
    int64_t D = 768;
    int64_t B = 1;
    std::vector<int64_t> seq_lens = {16, 32, 64, 128};

    for (int64_t S : seq_lens) {
        quant::FusionConfig cfg;
        cfg.hidden_size = D;
        cfg.num_heads = 8;
        cfg.head_dim = 96;
        cfg.num_fusion_layers = 2;
        cfg.dropout = 0.0f;

        quant::MultimodalFusion fusion(cfg);

        auto vision = make_random(B, S, D);
        auto audio = make_random(B, S, D);
        auto text = make_random(B, S, D);

        int warmup = 5;
        for (int i = 0; i < warmup; i++)
            fusion.fuse_all(vision, audio, text);

        int iters = 20;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            fusion.fuse_all(vision, audio, text);
        double dt = (now_sec() - t0) / iters;

        std::cout << "  S=" << std::setw(4) << S
                  << "  " << std::fixed << std::setprecision(3) << dt * 1000 << " ms/fusion\n";
    }
}

static void bench_cross_modal_attention() {
    std::cout << "\n=== Cross-Modal Attention Latency Breakdown ===\n";
    int64_t D = 768;
    int64_t B = 1;

    struct BenchCase { std::string name; int64_t Q; int64_t KV; };
    std::vector<BenchCase> cases = {
        {"vision→text",    64,  64},
        {"vision→audio",   64, 128},
        {"text→audio",    128, 128},
        {"vision→∞",       64, 256},
        {"all→∞",         256, 256},
    };

    for (const auto& c : cases) {
        quant::CrossAttentionFusion attn(D);
        auto query = make_random(B, c.Q, D);
        auto kv = make_random(B, c.KV, D);

        int warmup = 5;
        for (int i = 0; i < warmup; i++)
            attn.forward(query, kv);

        int iters = 30;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            attn.forward(query, kv);
        double dt = (now_sec() - t0) / iters * 1e6;

        std::cout << "  " << std::setw(16) << c.name
                  << "  Q=" << std::setw(4) << c.Q << " KV=" << std::setw(4) << c.KV
                  << "  " << std::fixed << std::setprecision(1) << dt << " us\n";
    }
}

int main() {
    std::cout << "=== QUANT Multimodal Benchmarks ===\n";

    bench_vision_encoder();
    bench_audio_encoder();
    bench_multimodal_fusion();
    bench_cross_modal_attention();

    std::cout << "\nDone.\n";
    return 0;
}
