#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/optimizer.h"
#include "quant/tensor.h"
#include "quant/math.h"
#include "quant/autograd.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <numeric>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

static int64_t count_params(quant::DenseModel& model) {
    return model.param_count();
}

static void bench_forward_throughput() {
    std::cout << "\n=== Forward Throughput ===\n";
    std::vector<quant::DenseModel> models;
    std::vector<int64_t> sizes = {128, 256, 512};
    int64_t seq_len = 128;

    for (int64_t hidden : sizes) {
        quant::TransformerConfig cfg;
        cfg.hidden_size = hidden;
        cfg.num_layers = 4;
        cfg.num_heads = hidden / 64;
        cfg.head_dim = 64;
        cfg.ffn_hidden_size = hidden * 4;
        cfg.vocab_size = 10000;
        cfg.max_seq_len = seq_len;
        cfg.activation = quant::Activation::SiLU;
        quant::DenseModel model(cfg);
        int64_t params = count_params(model);

        quant::Tensor input({1, seq_len}, quant::DType::F32);
        quant::Tensor pos({1, seq_len}, quant::DType::F32);
        float* id = input.data<float>();
        float* pd = pos.data<float>();
        for (int64_t i = 0; i < seq_len; i++) {
            id[i] = (float)(i % cfg.vocab_size);
            pd[i] = (float)i;
        }

        int warmup = 5;
        for (int i = 0; i < warmup; i++)
            model.forward(input, pos);

        int iters = 20;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            model.forward(input, pos);
        double dt = (now_sec() - t0) / iters;

        double tokens_per_sec = seq_len / dt;
        double tflop = 2.0 * (double)params * (double)seq_len / 1e12;
        double util = tflop / dt;

        std::cout << "  hidden=" << std::setw(4) << hidden
                  << "  params=" << std::setw(8) << params
                  << "  " << std::fixed << std::setprecision(0) << tokens_per_sec << " tok/s"
                  << "  " << std::fixed << std::setprecision(3) << dt * 1000 << " ms/step"
                  << "  " << std::fixed << std::setprecision(2) << util << " TFLOP/s\n";
    }
}

static void bench_backward_throughput() {
    std::cout << "\n=== Forward+Backward Throughput (Training Step) ===\n";
    int64_t hidden = 256;
    int64_t seq_len = 64;
    int64_t batch_size = 4;

    quant::TransformerConfig cfg;
    cfg.hidden_size = hidden;
    cfg.num_layers = 4;
    cfg.num_heads = hidden / 64;
    cfg.head_dim = 64;
    cfg.ffn_hidden_size = hidden * 4;
    cfg.vocab_size = 10000;
    cfg.max_seq_len = seq_len;
    cfg.activation = quant::Activation::SiLU;
    quant::DenseModel model(cfg);

    int64_t total_tokens = batch_size * seq_len;
    quant::Tensor input({batch_size, seq_len}, quant::DType::F32);
    quant::Tensor pos({batch_size, seq_len}, quant::DType::F32);
    float* id = input.data<float>();
    float* pd = pos.data<float>();
    for (int64_t i = 0; i < total_tokens; i++) {
        id[i] = (float)(i % cfg.vocab_size);
        pd[i] = (float)(i % seq_len);
    }

    quant::Tensor labels({batch_size * seq_len}, quant::DType::F32);
    float* ld = labels.data<float>();
    for (int64_t i = 0; i < total_tokens; i++)
        ld[i] = (float)((i + 1) % cfg.vocab_size);

    int warmup = 3;
    for (int i = 0; i < warmup; i++) {
        quant::Tensor logits = model.forward(input, pos);
        quant::Tensor loss = quant::cross_entropy_loss(logits, labels);
        quant::AutogradEngine::instance().backward(loss);
        quant::AutogradEngine::instance().clear();
    }

    int iters = 10;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        quant::Tensor logits = model.forward(input, pos);
        quant::Tensor loss = quant::cross_entropy_loss(logits, labels);
        quant::AutogradEngine::instance().backward(loss);
        quant::AutogradEngine::instance().clear();
    }
    double dt = (now_sec() - t0) / iters;

    double tokens_per_sec = total_tokens / dt;
    std::cout << "  batch=" << batch_size << " seq=" << seq_len
              << " total_tok=" << total_tokens
              << "  " << std::fixed << std::setprecision(0) << tokens_per_sec << " tok/s"
              << "  " << std::fixed << std::setprecision(3) << dt * 1000 << " ms/step\n";
}

static void bench_batch_size_scaling() {
    std::cout << "\n=== Batch Size Scaling ===\n";
    int64_t hidden = 192;
    int64_t seq_len = 64;

    quant::TransformerConfig cfg;
    cfg.hidden_size = hidden;
    cfg.num_layers = 3;
    cfg.num_heads = hidden / 64;
    cfg.head_dim = 64;
    cfg.ffn_hidden_size = hidden * 4;
    cfg.vocab_size = 10000;
    cfg.max_seq_len = seq_len;
    cfg.activation = quant::Activation::SiLU;

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16};
    for (int bs : batch_sizes) {
        quant::DenseModel model(cfg);
        int64_t total_tokens = bs * seq_len;
        quant::Tensor input({bs, seq_len}, quant::DType::F32);
        quant::Tensor pos({bs, seq_len}, quant::DType::F32);
        float* id = input.data<float>();
        float* pd = pos.data<float>();
        for (int64_t i = 0; i < total_tokens; i++) {
            id[i] = (float)(i % cfg.vocab_size);
            pd[i] = (float)(i % seq_len);
        }

        int warmup = 3;
        for (int i = 0; i < warmup; i++) model.forward(input, pos);

        int iters = 20;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) model.forward(input, pos);
        double dt = (now_sec() - t0) / iters;

        double tok_s = total_tokens / dt;
        std::cout << "  batch_size=" << std::setw(2) << bs
                  << "  " << std::fixed << std::setprecision(0) << tok_s << " tok/s"
                  << "  " << std::fixed << std::setprecision(3) << dt * 1000 << " ms\n";
    }
}

static void bench_gradient_compute() {
    std::cout << "\n=== Gradient Computation Breakdown ===\n";
    int64_t hidden = 256;
    int64_t seq_len = 64;
    int64_t batch_size = 4;

    quant::TransformerConfig cfg;
    cfg.hidden_size = hidden;
    cfg.num_layers = 4;
    cfg.num_heads = hidden / 64;
    cfg.head_dim = 64;
    cfg.ffn_hidden_size = hidden * 4;
    cfg.vocab_size = 10000;
    cfg.max_seq_len = seq_len;
    cfg.activation = quant::Activation::SiLU;

    quant::DenseModel model(cfg);
    int64_t total_tokens = batch_size * seq_len;

    quant::Tensor input({batch_size, seq_len}, quant::DType::F32);
    quant::Tensor pos({batch_size, seq_len}, quant::DType::F32);
    float* id = input.data<float>();
    float* pd = pos.data<float>();
    for (int64_t i = 0; i < total_tokens; i++) {
        id[i] = (float)(i % cfg.vocab_size);
        pd[i] = (float)(i % seq_len);
    }

    quant::Tensor labels({total_tokens}, quant::DType::F32);
    float* ld = labels.data<float>();
    for (int64_t i = 0; i < total_tokens; i++)
        ld[i] = (float)((i + 1) % cfg.vocab_size);

    // Forward-only
    int iters = 10;
    for (int w = 0; w < 3; w++) model.forward(input, pos);
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) model.forward(input, pos);
    double fwd = (now_sec() - t0) / iters;

    // Forward + backward + zero_grad
    for (int w = 0; w < 3; w++) {
        quant::Tensor logits = model.forward(input, pos);
        quant::Tensor loss = quant::cross_entropy_loss(logits, labels);
        quant::AutogradEngine::instance().backward(loss);
        quant::AutogradEngine::instance().clear();
    }
    t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        quant::Tensor logits = model.forward(input, pos);
        quant::Tensor loss = quant::cross_entropy_loss(logits, labels);
        quant::AutogradEngine::instance().backward(loss);
        quant::AutogradEngine::instance().clear();
    }
    double full = (now_sec() - t0) / iters;

    std::cout << "  Forward:        " << std::fixed << std::setprecision(3) << fwd * 1000 << " ms\n";
    std::cout << "  Forward+Backward: " << std::fixed << std::setprecision(3) << full * 1000 << " ms\n";
    std::cout << "  Backward only:   " << std::fixed << std::setprecision(3) << (full - fwd) * 1000 << " ms\n";
    std::cout << "  Forward ratio:   " << std::fixed << std::setprecision(2) << (fwd / full * 100) << "%\n";
    std::cout << "  Backward ratio:  " << std::fixed << std::setprecision(2) << ((full - fwd) / full * 100) << "%\n";
}

int main() {
    std::cout << "=== QUANT Training Benchmarks ===\n";

    bench_forward_throughput();
    bench_backward_throughput();
    bench_batch_size_scaling();
    bench_gradient_compute();

    std::cout << "\nDone.\n";
    return 0;
}
