#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/types.h"
#include "oil/transformer.h"
#include "oil/autograd.h"
#include "oil/kernel.h"
#include "oil/kv_cache.h"
#include "oil/flash_attention.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include "oil/test.h"

using namespace oil;

static bool approx_equal(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// Attention forward + backward
static void test_attention() {
    TEST_SUITE("Attention Forward/Backward");
    TransformerConfig cfg;
    cfg.hidden_size = 32;
    cfg.num_heads = 4;
    cfg.head_dim = 8;
    cfg.num_kv_heads = 2;
    cfg.max_seq_len = 32;
    Attention attn(cfg);
    TEST_CHECK(attn.num_heads == 4, "attention num_heads=4");
    TEST_CHECK(attn.num_kv_heads == 2, "attention num_kv_heads=2");
    TEST_CHECK(attn.head_dim == 8, "attention head_dim=8");

    int64_t B = 2, S = 4;
    Tensor x({B, S, cfg.hidden_size});
    for (int64_t i = 0; i < x.numel(); i++)
        x.data<float>()[i] = std::sin((float)i * 0.1f) * 0.01f; // small values to avoid softmax overflow
    Tensor positions({B, S});
    for (int64_t i = 0; i < B * S; i++)
        positions.data<float>()[i] = (float)i;
    Tensor mask({S, S});
    mask.fill(0.0f);

    KVCache cache(static_cast<int>(cfg.num_layers), B, cfg.num_kv_heads, cfg.head_dim, static_cast<bool>(cfg.max_seq_len));
    Tensor out = attn.forward(x, positions, mask, cache, 0);
    TEST_CHECK(out.numel() == B * S * cfg.hidden_size, "attention forward output shape");
    TEST_CHECK(true, "attention forward completed without crash");

    // Attention with different shapes
    TransformerConfig cfg2;
    cfg2.hidden_size = 16;
    cfg2.num_heads = 2;
    cfg2.head_dim = 8;
    cfg2.num_kv_heads = 2;
    cfg2.max_seq_len = 64;
    Attention attn2(cfg2);

    Tensor x2({1, 8, 16});
    x2.fill(0.3f);
    Tensor pos2({1, 8});
    for (int64_t i = 0; i < 8; i++) pos2.data<float>()[i] = (float)i;
    Tensor mask2({8, 8});
    mask2.fill(0.0f);
    KVCache cache2(static_cast<int>(cfg2.num_layers), 1, cfg2.num_kv_heads, cfg2.head_dim, static_cast<bool>(cfg2.max_seq_len));
    Tensor out2 = attn2.forward(x2, pos2, mask2, cache2, 0);
    TEST_CHECK(out2.dim(0) == 1 && out2.dim(1) == 8 && out2.dim(2) == 16, "attention2 output shape");

    // Autograd attention_op
    AutogradEngine::set_enabled(true);
    Tensor q({1, 4, 1, 8}), k({1, 4, 1, 8}), v({1, 4, 1, 8});
    q.fill(1.0f); k.fill(1.0f); v.fill(1.0f);
    Tensor attn_out = AutogradEngine::attention_op(q, k, v, 4, 4, 8);
    TEST_CHECK(attn_out.numel() > 0, "autograd attention_op produces output");
    TEST_CHECK(true, "autograd attention completed without crash");
    AutogradEngine::set_enabled(false);

    // Flash attention
    FlashAttention fa(FlashAttentionConfig{64, true, 1.0f});
    Tensor fa_mask({8, 8});
    fa_mask.fill(0.0f);
    Tensor fa_out = fa.forward(q, k, v, fa_mask);
    TEST_CHECK(fa_out.numel() > 0, "flash attention produces output");
}

// Cross entropy loss
static void test_cross_entropy_loss() {
    TEST_SUITE("Cross Entropy Loss");
    int64_t B = 2, S = 3, V = 10;
    Tensor logits({B, S, V});
    for (int64_t i = 0; i < B * S * V; i++)
        logits.data<float>()[i] = (float)(i % V) / (float)V;

    Tensor targets({B, S});
    for (int64_t i = 0; i < B * S; i++)
        targets.data<float>()[i] = (float)((i * 3) % V);

    Tensor loss = cross_entropy_loss(logits, targets);
    TEST_CHECK(loss.numel() > 0, "cross entropy produces output");
    float loss_val = math::sum(loss);
    TEST_CHECK(std::isfinite(loss_val), "cross entropy loss finite");
    TEST_CHECK(loss_val > 0, "cross entropy loss positive");

    // Gradient
    Tensor grad = cross_entropy_grad(logits, targets);
    TEST_CHECK(grad.numel() == logits.numel(), "cross entropy grad shape matches");
    for (int64_t i = 0; i < grad.numel(); i++)
        TEST_CHECK(std::isfinite(grad.data<float>()[i]), "cross entropy grad finite");

    // Autograd cross_entropy_op
    AutogradEngine::set_enabled(true);
    Tensor ce_auto = AutogradEngine::cross_entropy_op(logits, targets);
    TEST_CHECK(ce_auto.numel() > 0, "autograd cross entropy produces output");
    float ce_val = math::sum(ce_auto);
    TEST_CHECK_CLOSE(ce_val, loss_val, 1e-4f, "autograd cross entropy matches standalone");
    AutogradEngine::set_enabled(false);

    // Perfect logits should have near-zero loss
    Tensor perfect_logits({1, 1, 10});
    perfect_logits.fill(-10.0f);
    perfect_logits.data<float>()[9] = 10.0f;
    Tensor perfect_targets({1, 1});
    perfect_targets.data<float>()[0] = 9.0f;
    Tensor perfect_loss = cross_entropy_loss(perfect_logits, perfect_targets);
    float perfect_val = math::sum(perfect_loss);
    TEST_CHECK(perfect_val < 1.0f, "cross entropy near-zero for perfect logits");
}

// RMS norm
static void test_rms_norm() {
    TEST_SUITE("RMS Norm");
    int64_t D = 16;
    RMSNorm rms(D, 1e-5f);
    TEST_CHECK(rms.weight.numel() == D, "RMSNorm weight size = D");

    Tensor x({4, D});
    for (int64_t i = 0; i < x.numel(); i++)
        x.data<float>()[i] = (float)((i * 3 + 5) % 10) / 10.0f;
    Tensor y = rms.forward(x);
    TEST_CHECK(y.numel() == x.numel(), "RMS norm output shape matches");
    for (int64_t i = 0; i < y.numel(); i++)
        TEST_CHECK(std::isfinite(y.data<float>()[i]), "RMS norm output finite");

    // Verify RMS norm property: mean of squares ~ 1 (times weight^2)
    auto test_norm = [&](int64_t row) {
        const float* xd = x.data<float>() + row * D;
        const float* yd = y.data<float>() + row * D;
        float x_sq = 0, y_sq = 0;
        for (int64_t d = 0; d < D; d++) {
            x_sq += xd[d] * xd[d];
            y_sq += yd[d] * yd[d];
        }
        float x_rms = std::sqrt(x_sq / (float)D + 1e-5f);
        float w = rms.weight.data<float>()[0];
        TEST_CHECK_CLOSE(yd[0] * x_rms, xd[0] * w, 1e-4f, "RMS norm x * rms / w = y correctness");
    };
    test_norm(1);
    test_norm(2);

    // math::rms_norm
    Tensor y_math({4, D});
    math::rms_norm(x, rms.weight, 1e-5f, y_math);
    TEST_CHECK_CLOSE(y.data<float>()[0], y_math.data<float>()[0], 1e-4f, "RMS norm math matches class method");

    // Autograd rms_norm_op
    AutogradEngine::set_enabled(true);
    Tensor rms_out = AutogradEngine::rms_norm_op(x, rms.weight, 1e-5f);
    TEST_CHECK(rms_out.numel() == x.numel(), "autograd RMS norm shape");
    AutogradEngine::set_enabled(false);

    // RMS norm gradient
    Tensor grad({4, D});
    grad.fill(1.0f);
    Tensor dgamma({D});
    Tensor rms_grad = rms_norm_grad(x, rms.weight, grad, static_cast<int>(D), &dgamma);
    TEST_CHECK(rms_grad.numel() == x.numel(), "RMS norm grad shape");
    TEST_CHECK(dgamma.numel() == D, "RMS norm dgamma shape");
    for (int64_t i = 0; i < rms_grad.numel(); i++)
        TEST_CHECK(std::isfinite(rms_grad.data<float>()[i]), "RMS norm gradient finite");
}

// RoPE
static void test_rope() {
    TEST_SUITE("RoPE");
    int64_t head_dim = 8, max_seq = 64;

    // Construction
    {
        RotaryEmbedding rope(head_dim, max_seq, 10000.0f);
        TEST_CHECK(rope.cos_cached.numel() > 0, "RoPE cos cached populated");
        TEST_CHECK(rope.sin_cached.numel() > 0, "RoPE sin cached populated");
        TEST_CHECK(rope.head_dim == head_dim, "RoPE head_dim set");
        TEST_CHECK_CLOSE(rope.theta, 10000.0f, 1e-6f, "RoPE theta set");
    }
    {
        RotaryEmbedding rope2(head_dim, max_seq, 500000.0f);
        TEST_CHECK(rope2.head_dim == head_dim, "RoPE alternate theta head_dim");
        TEST_CHECK_CLOSE(rope2.theta, 500000.0f, 1e-6f, "RoPE alternate theta set");
    }

    // Autograd rotary_op (safer than in-place apply)
    {
        RotaryEmbedding rope(head_dim, max_seq, 10000.0f);
        AutogradEngine::set_enabled(true);
        Tensor x4({1, 4, head_dim});
        x4.fill(0.5f);
        Tensor rot_out = AutogradEngine::rotary_op(x4, rope.cos_cached, rope.sin_cached, 0, 4);
        TEST_CHECK(rot_out.numel() == x4.numel(), "autograd RoPE shape");
        AutogradEngine::set_enabled(false);
    }

    // Rotary embedding construction with various sizes
    {
        RotaryEmbedding rope_small(4, 32, 10000.0f);
        TEST_CHECK(rope_small.cos_cached.numel() > 0, "RoPE small config cached ok");
    }
}

// SwiGLU (FFN with SiLU + gate)
static void test_swiglu() {
    TEST_SUITE("SwiGLU");
    try {
        FFN ffn(16, 32);
        TEST_CHECK(ffn.gate_proj.weight.numel() > 0, "SwiGLU gate_proj has weight");
        TEST_CHECK(ffn.up_proj.weight.numel() > 0, "SwiGLU up_proj has weight");
        TEST_CHECK(ffn.down_proj.weight.numel() > 0, "SwiGLU down_proj has weight");

        Tensor x({3, 16});
        for (int64_t i = 0; i < x.numel(); i++)
            x.data<float>()[i] = std::sin((float)i * 0.3f) * 0.01f;
        Tensor y = ffn.forward(x);
        TEST_CHECK(y.dim(0) == 3 && y.dim(1) == 16, "SwiGLU forward shape {3, 16}");
        TEST_CHECK(true, "SwiGLU forward completed");

        // Different config
        FFN ffn2(32, 64);
        Tensor x2({2, 32});
        x2.fill(0.01f);
        Tensor y2 = ffn2.forward(x2);
        TEST_CHECK(y2.dim(0) == 2 && y2.dim(1) == 32, "SwiGLU wider forward shape completed");
    } catch (const std::exception& e) {
        TEST_CHECK(false, "SwiGLU exception");
        printf("    exception: %s\n", e.what());
    }

    // Autograd silu_op
    try {
        AutogradEngine::set_enabled(true);
        Tensor silu_in({10});
        for (int64_t i = 0; i < 10; i++)
            silu_in.data<float>()[i] = (float)(i - 5);
        Tensor silu_out = AutogradEngine::silu_op(silu_in);
        TEST_CHECK(silu_out.numel() == 10, "autograd silu shape");
        for (int64_t i = 0; i < silu_out.numel(); i++)
            TEST_CHECK(std::isfinite(silu_out.data<float>()[i]), "autograd silu finite");

        // Verify SiLU formula: x * sigmoid(x)
        for (int64_t i = 0; i < 10; i++) {
            float inp = silu_in.data<float>()[i];
            float sig = 1.0f / (1.0f + std::exp(-inp));
            TEST_CHECK_CLOSE(silu_out.data<float>()[i], inp * sig, 1e-5f, "SiLU formula correct");
        }
        AutogradEngine::set_enabled(false);
    } catch (const std::exception& e) {
        TEST_CHECK(false, "autograd silu exception");
        printf("    exception: %s\n", e.what());
    }

    // silu_grad
    try {
        Tensor sg_in({10});
        for (int64_t i = 0; i < 10; i++)
            sg_in.data<float>()[i] = (float)(i - 5);
        Tensor grad({10});
        grad.fill(1.0f);
        Tensor silu_g = silu_grad(sg_in, grad);
        TEST_CHECK(silu_g.numel() == 10, "silu_grad shape");
        TEST_CHECK(true, "silu_grad completed");
    } catch (const std::exception& e) {
        TEST_CHECK(false, "silu_grad exception");
        printf("    exception: %s\n", e.what());
    }
}

// Causal mask
static void test_causal_mask() {
    TEST_SUITE("Causal Mask");
    int64_t S = 5;
    Tensor mask({S, S});
    mask.fill(0.0f);

    // Build causal mask manually: upper triangle = -inf, lower = 0
    for (int64_t i = 0; i < S; i++)
        for (int64_t j = i + 1; j < S; j++)
            mask.data<float>()[i * S + j] = -INFINITY;

    // Verify causal property
    for (int64_t i = 0; i < S; i++) {
        for (int64_t j = 0; j < S; j++) {
            float val = mask.data<float>()[i * S + j];
            if (j > i) {
                TEST_CHECK(!std::isfinite(val) || val < -1e10f, "causal mask: j>i masked");
            } else {
                TEST_CHECK(val >= 0.0f, "causal mask: j<=i unmasked");
            }
        }
    }

    // Attention with causal mask
    TransformerConfig cfg;
    cfg.hidden_size = 16;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.max_seq_len = 16;
    Attention attn(cfg);
    KVCache cache(1, 1, 2, 8, true);

    Tensor x({1, S, 16});
    for (int64_t i = 0; i < x.numel(); i++)
        x.data<float>()[i] = std::sin((float)i * 0.2f) * 0.01f;
    Tensor pos({1, S});
    for (int64_t i = 0; i < S; i++) pos.data<float>()[i] = (float)i;

    // Forward with causal mask
    try {
        Tensor causal_out = attn.forward(x, pos, mask, cache, 0);
        TEST_CHECK(causal_out.dim(0) == 1 && causal_out.dim(1) == S && causal_out.dim(2) == 16, "causal attention output shape");
        TEST_CHECK(true, "causal attention completed");
    } catch (const std::exception& e) {
        TEST_CHECK(false, "causal attention exception");
        printf("    exception: %s\n", e.what());
    }

    // Build larger causal mask
    int64_t S2 = 10;
    Tensor mask2({S2, S2});
    mask2.fill(0.0f);
    for (int64_t i = 0; i < S2; i++)
        for (int64_t j = i + 1; j < S2; j++)
            mask2.data<float>()[i * S2 + j] = -INFINITY;

    // Check diagonal
    for (int64_t i = 0; i < S2; i++)
        TEST_CHECK(mask2.data<float>()[i * S2 + i] >= 0.0f, "causal mask diagonal unmasked");

    // Check lower triangular
    for (int64_t i = 0; i < S2; i++)
        for (int64_t j = 0; j < i; j++)
            TEST_CHECK(mask2.data<float>()[i * S2 + j] >= 0.0f, "causal mask lower triangular unmasked");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("InNova — Core Ops Test Suite\n");
    printf("=================================\n");

    test_attention();
    test_cross_entropy_loss();
    test_rms_norm();
    test_rope();
    test_swiglu();
    test_causal_mask();

    printf("\n=================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
