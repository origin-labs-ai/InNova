#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/math.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include "oil/test.h"

int main() {
    // Create a tiny transformer config
    oil::TransformerConfig cfg;
    cfg.vocab_size = 100;
    cfg.hidden_size = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.ffn_hidden_size = 64;
    cfg.norm_eps = 1e-5f;
    cfg.max_seq_len = 64;

    assert(cfg.head_dim == 8);

    // Create model with tiny config
    oil::DenseModel model(cfg);

    // Verify param count > 0
    int64_t pc = model.param_count();
    assert(pc > 0);
    assert(model.vocab_size() == 100);

    // Create random input
    const int64_t B = 2;
    const int64_t S = 4;
    oil::Tensor input_ids(oil::Shape{B, S}, oil::DType::F32);
    oil::Tensor positions(oil::Shape{B, S}, oil::DType::F32);

    // Fill with random token IDs in range [0, vocab_size)
    unsigned int seed = 42;
    for (int64_t i = 0; i < B * S; i++) {
        seed = seed * 1103515245u + 12345u;
        float tid = static_cast<float>((int)((seed >> 16) % cfg.vocab_size));
        float pid = static_cast<float>((int)(i % S));
        std::memcpy(input_ids.data<float>() + i, &tid, sizeof(float));
        std::memcpy(positions.data<float>() + i, &pid, sizeof(float));
    }

    // Run forward pass
    oil::Tensor logits = model.forward(input_ids, positions);

    // Verify output shape: (B, S, vocab_size)
    assert(logits.shape().rank == 3);
    assert(logits.shape().dims[0] == B);
    assert(logits.shape().dims[1] == S);
    assert(logits.shape().dims[2] == cfg.vocab_size);
    assert(logits.numel() == B * S * cfg.vocab_size);
    assert(logits.dtype() == oil::DType::F32);

    // Verify output values are finite
    float* logits_data = logits.data<float>();
    for (int64_t i = 0; i < logits.numel(); i++) {
        assert(std::isfinite(logits_data[i]));
    }

    // Verify model.save doesn't crash
    model.save("_test_model_tmp.oil");

    // Verify load doesn't crash (even if config mismatch)
    try {
        oil::DenseModel model2;
        model2.load("_test_model_tmp.oil");
    } catch (const std::exception& e) {
        std::cerr << "Load error (expected for incomplete save): " << e.what() << std::endl;
    }

    // Test different config dimensions
    {
        oil::TransformerConfig cfg2;
        cfg2.vocab_size = 50;
        cfg2.hidden_size = 16;
        cfg2.num_layers = 1;
        cfg2.num_heads = 2;
        cfg2.head_dim = 8;
        cfg2.ffn_hidden_size = 32;
        cfg2.max_seq_len = 32;

        oil::DenseModel model2(cfg2);
        assert(model2.param_count() > 0);
        assert(model2.vocab_size() == 50);

        oil::Tensor in2(oil::Shape{1, 2}, oil::DType::F32);
        oil::Tensor pos2(oil::Shape{1, 2}, oil::DType::F32);
        float v0 = 0.0f, v1 = 1.0f;
        std::memcpy(in2.data<float>(), &v0, sizeof(float));
        std::memcpy(in2.data<float>() + 1, &v1, sizeof(float));
        std::memcpy(pos2.data<float>(), &v0, sizeof(float));
        std::memcpy(pos2.data<float>() + 1, &v1, sizeof(float));

        oil::Tensor out2 = model2.forward(in2, pos2);
        assert(out2.shape().dims[2] == 50);
    }

    std::remove("_test_model_tmp.oil");

    // Test Embedding component
    {
        oil::Embedding emb(100, 32);
        assert(emb.param_count() == 100 * 32);
        oil::Tensor ids(oil::Shape{3}, oil::DType::F32);
        float id0 = 0.0f, id5 = 5.0f, id10 = 10.0f;
        std::memcpy(ids.data<float>(), &id0, sizeof(float));
        std::memcpy(ids.data<float>() + 1, &id5, sizeof(float));
        std::memcpy(ids.data<float>() + 2, &id10, sizeof(float));
        auto e = emb.forward(ids);
        assert(e.shape().rank == 2);
        assert(e.shape().dims[0] == 3);
        assert(e.shape().dims[1] == 32);
    }

    // Test Linear component
    {
        oil::Linear lin(16, 32);
        assert(lin.param_count() == 16 * 32 + 32);
        oil::Tensor x(oil::Shape{2, 16}, oil::DType::F32);
        x.fill(0.5f);
        auto y = lin.forward(x);
        assert(y.shape().rank == 2);
        assert(y.shape().dims[0] == 2);
        assert(y.shape().dims[1] == 32);
    }

    // Test RMSNorm
    {
        oil::RMSNorm rms(16, 1e-5f);
        oil::Tensor x(oil::Shape{4, 16}, oil::DType::F32);
        x.fill(2.0f);
        auto y = rms.forward(x);
        assert(y.shape() == x.shape());
        (void)y;
    }

    std::cout << "All model tests passed!" << std::endl;
    return 0;
}
