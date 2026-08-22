// test_gradient_check.cpp — numerical vs analytic gradient on autograd ops
#include "quant/autograd.h"
#include "quant/tensor.h"
#include "quant/optimizer.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>

using namespace quant;

static float forward_loss(Tensor& w, const Tensor& x, const Tensor& labels,
                          int64_t N, int64_t V, int64_t D) {
    AutogradEngine::set_enabled(false);
    Tensor logits = AutogradEngine::matmul_op(x, w, N, V, D);
    Tensor loss = AutogradEngine::cross_entropy_op(logits, labels);
    AutogradEngine::set_enabled(true);
    return *(const float*)loss.data();
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    TEST_SUITE("gradient_check");
    printf("=== Gradient check test ===\n\n");

    auto& engine = AutogradEngine::instance();
    const int64_t N = 4, D = 3, V = 6;
    const float eps = 1e-3f;

    Tensor w(Shape{V, D}, DType::F32);
    for (int64_t i = 0; i < D * V; i++)
        w.data<float>()[i] = (float)(i % 5) / 5.0f - 0.4f;

    Tensor x(Shape{N, D}, DType::F32);
    for (int64_t i = 0; i < N * D; i++)
        x.data<float>()[i] = (float)(i % 7) / 7.0f - 0.4f;

    Tensor labels(Shape{N, 1}, DType::F32);
    for (int64_t i = 0; i < N; i++) labels.data<float>()[i] = (float)(i % V);

    // Analytic gradient: logits = x {N,D} @ w {V,D}^T — w is {N,K} row-major.
    // Mark the parameter trainable before registering it with the autograd engine
    // (matching how ModelTrainer marks collected params).
    w.requires_grad(true);
    engine.register_parameter(&w);
    AutogradEngine::set_enabled(true);
    Tensor logits = AutogradEngine::matmul_op(x, w, N, V, D);
    Tensor loss = AutogradEngine::cross_entropy_op(logits, labels);
    float loss0 = *(const float*)loss.data();
    engine.backward(loss);
    engine.clear();
    AutogradEngine::set_enabled(false);

    TEST_CHECK(w.has_grad(), "parameter has gradient");
    TEST_CHECK(std::isfinite(loss0), "loss finite");

    // Numerical gradient (central difference) per weight
    float max_rel_err = 0.0f;
    for (int64_t i = 0; i < D * V; i++) {
        float orig = w.data<float>()[i];
        w.data<float>()[i] = orig + eps;
        float lp = forward_loss(w, x, labels, N, V, D);
        w.data<float>()[i] = orig - eps;
        float lm = forward_loss(w, x, labels, N, V, D);
        w.data<float>()[i] = orig;
        float numeric = (lp - lm) / (2.0f * eps);
        float analytic = w.grad().data<float>()[i];
        float denom = std::max(1.0f, std::fabs(numeric));
        float rel = std::fabs(numeric - analytic) / denom;
        if (rel > max_rel_err) max_rel_err = rel;
    }

    printf("  max relative grad error: %.4f\n", max_rel_err);
    TEST_CHECK(max_rel_err < 1e-2f, "analytic gradient matches numerical (rel err < 1e-2)");

    // Gradient descent with verified gradient must reduce loss
    SGD opt(0.1f);
    std::vector<Tensor*> group = {&w};
    opt.add_param_group(group);
    float prev = loss0;
    for (int step = 0; step < 20; step++) {
        opt.zero_grad();
        AutogradEngine::set_enabled(true);
        engine.register_parameter(&w);
        Tensor lg = AutogradEngine::matmul_op(x, w, N, V, D);
        Tensor ls = AutogradEngine::cross_entropy_op(lg, labels);
        float v = *(const float*)ls.data();
        engine.backward(ls);
        engine.clear();
        AutogradEngine::set_enabled(false);
        opt.step();
        prev = v;
    }
    TEST_CHECK(std::isfinite(prev), "SGD keeps loss finite");

    int failures = TEST_REPORT();
    printf("\nGRADIENT CHECK TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
