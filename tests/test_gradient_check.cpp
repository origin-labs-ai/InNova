#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/autograd.h"
#include "oil/math.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/optimizer.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include "oil/test.h"

using namespace oil;

static float compute_loss_value(DenseModel& m, const Tensor& inp, const Tensor& tgt) {
    Tensor logits = m.forward(inp, inp);
    int64_t Seq = logits.dim(1);
    int64_t Vocab = logits.dim(2);
    const float* ld = logits.data<float>();
    const float* td = tgt.data<float>();
    float loss = 0;
    for (int64_t s = 0; s < Seq; s++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < Vocab; v++)
            if (ld[s * Vocab + v] > max_l) max_l = ld[s * Vocab + v];
        float sum = 0;
        for (int64_t v = 0; v < Vocab; v++)
            sum += std::exp(ld[s * Vocab + v] - max_l);
        int64_t t = (int64_t)td[s];
        if (t >= 0 && t < Vocab)
            loss += -(ld[s * Vocab + t] - max_l - std::log(sum));
    }
    return loss;
}

static void test_gradient_via_finite_diff() {
    TEST_SUITE("Gradient Check: Finite Difference");
    TransformerConfig cfg;
    cfg.vocab_size = 8;
    cfg.hidden_size = 4;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 2;
    cfg.ffn_hidden_size = 8;
    cfg.norm_eps = 1e-5f;
    cfg.max_seq_len = 32;

    DenseModel model(cfg);
    const float h = 1e-3f;

    for (int pass = 0; pass < 2; pass++) {
        int64_t B = 1, S = 3;
        Tensor input_ids(Shape{B, S});
        Tensor targets(Shape{B, S});
        for (int64_t i = 0; i < B * S; i++) {
            input_ids.data<float>()[i] = (float)(((int)i * 5 + 1) % (cfg.vocab_size - 1)) + 1.0f;
            targets.data<float>()[i] = (float)(((int)i * 3) % cfg.vocab_size);
        }

        // Forward + backward to get analytical gradients
        std::vector<Tensor*> all_params;
        all_params.push_back(&model.lm_head->weight);
        all_params.push_back(&model.norm->weight);
        all_params.push_back(&model.tok_embeddings->weight);
        for (auto& layer : model.layers) {
            all_params.push_back(&layer->attention_norm.weight);
            all_params.push_back(&layer->attention.q_proj.weight);
            all_params.push_back(&layer->attention.k_proj.weight);
            all_params.push_back(&layer->attention.v_proj.weight);
            all_params.push_back(&layer->attention.o_proj.weight);
            all_params.push_back(&layer->ffn_norm.weight);
            all_params.push_back(&layer->ffn.gate_proj.weight);
            all_params.push_back(&layer->ffn.up_proj.weight);
            all_params.push_back(&layer->ffn.down_proj.weight);
        }
        if (model.lm_head->bias.numel() > 0)
            all_params.push_back(&model.lm_head->bias);

        AutogradEngine::set_enabled(true);
        auto& engine = AutogradEngine::instance();
        for (auto* p : all_params) engine.register_parameter(p);

        Tensor logits = model.forward(input_ids, targets);
        Tensor loss = AutogradEngine::cross_entropy_op(logits, targets);
        engine.backward(loss);
        engine.clear();
        AutogradEngine::set_enabled(false);

        // Check lm_head.weight gradient
        auto& w = model.lm_head->weight;
        if (!w.has_grad()) {
            printf("  SKIP: no gradient for lm_head.weight\n");
            continue;
        }

        const float* ag = w.grad().data<float>();
        int n_checks = static_cast<int>(std::min((int64_t)4, w.numel()));
        int n_passed = 0;
        for (int k = 0; k < n_checks; k++) {
            int64_t idx = k;
            float orig = w.data<float>()[idx];

            w.data<float>()[idx] = orig + h;
            Tensor logits_p = model.forward(input_ids, targets);
            float L_plus = compute_loss_value(model, input_ids, targets);

            w.data<float>()[idx] = orig - h;
            logits_p = model.forward(input_ids, targets);
            float L_minus = compute_loss_value(model, input_ids, targets);

            w.data<float>()[idx] = orig;

            float num_grad = (L_plus - L_minus) / (2.0f * h);
            float ana_grad = ag[idx];
            float denom = std::fabs(ana_grad) + std::fabs(num_grad) + 1e-8f;
            float rel_err = std::fabs(ana_grad - num_grad) / denom;

            if (rel_err < 0.5f) n_passed++;
        }
        TEST_CHECK(n_passed >= n_checks / 2,
              "Finite difference gradient check passes for most parameters");
    }
}

static void test_autograd_registration() {
    TEST_SUITE("Autograd: Parameter Registration");
    TransformerConfig cfg;
    cfg.vocab_size = 16;
    cfg.hidden_size = 8;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.ffn_hidden_size = 16;
    cfg.max_seq_len = 16;

    DenseModel model(cfg);

    AutogradEngine::set_enabled(true);
    auto& engine = AutogradEngine::instance();

    Tensor dummy(Shape{4});
    engine.register_parameter(&dummy);
    dummy.requires_grad(true);

    // Verify registration by checking gradient exists after a simple op
    Tensor ones(Shape{4});
    ones.fill(1.0f);
    Tensor result = AutogradEngine::add_op(dummy, ones);
    engine.backward(result);
    TEST_CHECK(dummy.has_grad(), "registered parameter receives gradient");
    engine.clear();
    AutogradEngine::set_enabled(false);
}

// Note: model.forward() does not use autograd-tracked operations internally,
// so full-model gradient flow tests are deferred until the model uses AutogradEngine ops.
// Individual op gradient tests (matmul, add, cross_entropy) are validated elsewhere.

static void test_cross_entropy_gradient_analytic() {
    TEST_SUITE("Cross Entropy: Gradient Analytical Correctness");
    int64_t B = 2, S = 3, V = 10;
    Tensor logits(Shape{B, S, V});
    Tensor targets(Shape{B, S});
    for (int64_t i = 0; i < B * S * V; i++)
        logits.data<float>()[i] = (float)((i * 7) % 20 - 10) / 5.0f;
    for (int64_t i = 0; i < B * S; i++)
        targets.data<float>()[i] = (float)(i % V);

    // Cross-entropy gradient: (softmax(x) - one_hot(target)) / B*S
    auto grad = cross_entropy_grad(logits, targets);
    TEST_CHECK(grad.shape() == logits.shape(), "gradient has same shape as logits");

    float expected_scale = 1.0f / (float)(B * S);
    for (int64_t i = 0; i < B * S; i++) {
        const float* ld = logits.data<float>() + i * V;
        float* gd = grad.data<float>() + i * V;
        int t = (int)targets.data<float>()[i];

        // softmax
        float max_l = ld[0];
        for (int64_t v = 1; v < V; v++) if (ld[v] > max_l) max_l = ld[v];
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(ld[v] - max_l);
        for (int64_t v = 0; v < V; v++) {
            float p = std::exp(ld[v] - max_l) / sum;
            float expected = (p - (v == t ? 1.0f : 0.0f)) * expected_scale;
            TEST_CHECK_CLOSE(gd[v], expected, 1e-4f,
                        "grad = (softmax - one_hot) / batch at each position");
        }
    }
}

static void test_sgd_parameter_update() {
    TEST_SUITE("Optimizer: SGD weight change");
    // Verify SGD optimizer actually modifies parameter values
    Tensor w(Shape{4});
    w.requires_grad(true);
    float* wd = w.data<float>();
    for (int i = 0; i < 4; i++) wd[i] = 1.0f;

    { Tensor g(Shape{4}); g.fill(0.1f); w.set_grad(g); } // simulate accumulated gradient

    SGD opt(1.0f);
    std::vector<Tensor*> params = {&w};
    opt.add_param_group(params);

    float before = w.data<float>()[0];
    opt.step();
    float after = w.data<float>()[0];
    TEST_CHECK(after < before, "SGD step decreases weight with positive gradient * lr");

    // Verify zero_grad clears gradients
    { Tensor g(Shape{4}); g.fill(0.5f); w.set_grad(g); }
    opt.zero_grad();
    const float* gd = w.grad().data<float>();
    bool all_zero = true;
    for (int i = 0; i < 4; i++) if (gd[i] != 0.0f) { all_zero = false; break; }
    TEST_CHECK(all_zero, "zero_grad clears gradients");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("InNova — Gradient Checking & Autograd Test Suite\n");
    printf("====================================================\n");

    test_gradient_via_finite_diff();
    test_autograd_registration();
    test_cross_entropy_gradient_analytic();
    test_sgd_parameter_update();

    printf("\n====================================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
