#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/types.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    // Test dot product
    {
        auto a = oil::Tensor::arange(5);
        auto b = oil::Tensor::arange(5);
        float d = oil::math::dot(a, b);
        (void)d;
        // 0*0 + 1*1 + 2*2 + 3*3 + 4*4 = 0+1+4+9+16 = 30
        assert(std::abs(d - 30.0f) < 1e-5f);
    }

    // Test axpy: y += alpha * x
    {
        auto x = oil::Tensor::ones({10});
        auto y = oil::Tensor::ones({10});
        oil::math::axpy(2.0f, x, y);
        for (int64_t i = 0; i < 10; i++)
            assert(std::abs(y.data<float>()[i] - 3.0f) < 1e-6f);
    }

    // Test norm
    {
        auto x = oil::Tensor::arange(4);
        // |x| = sqrt(0+1+4+9) = sqrt(14)
        float n = oil::math::norm(x); (void)n;
        assert(std::abs(n - std::sqrt(14.0f)) < 1e-5f);
    }

    // Test asum
    {
        auto x = oil::Tensor::arange(5);
        float s = oil::math::asum(x); (void)s;
        assert(std::abs(s - 10.0f) < 1e-5f);
    }

    // Test GEMM: verify A*B = expected result
    {
        // A = [[1,2],[3,4],[5,6]]  (3x2)
        // B = [[7,8,9],[10,11,12]] (2x3)
        // C = A*B -> (3x3)
        auto A = oil::Tensor::zeros({3, 2});
        auto B = oil::Tensor::zeros({2, 3});
        auto C = oil::Tensor::zeros({3, 3});
        float ad[] = {1,2,3,4,5,6};
        float bd[] = {7,8,9,10,11,12};
        memcpy(A.data(), ad, 6 * sizeof(float));
        memcpy(B.data(), bd, 6 * sizeof(float));

        oil::math::gemm(1.0f, A, B, 0.0f, C);

        // C[0][0] = 1*7 + 2*10 = 27
        // C[0][1] = 1*8 + 2*11 = 30
        // C[0][2] = 1*9 + 2*12 = 33
        // C[1][0] = 3*7 + 4*10 = 61
        // C[1][1] = 3*8 + 4*11 = 68
        // C[1][2] = 3*9 + 4*12 = 75
        // C[2][0] = 5*7 + 6*10 = 95
        // C[2][1] = 5*8 + 6*11 = 106
        // C[2][2] = 5*9 + 6*12 = 117
        float expected[] = {27,30,33,61,68,75,95,106,117}; (void)expected;
        for (int i = 0; i < 9; i++)
            assert(std::abs(C.data<float>()[i] - expected[i]) < 1e-5f);
    }

    // Test GEMV
    {
        auto A = oil::Tensor::zeros({2, 3});
        auto x = oil::Tensor::zeros({3});
        auto y = oil::Tensor::zeros({2});
        float ad[] = {1,2,3,4,5,6};
        float xd[] = {2,3,4};
        memcpy(A.data(), ad, 6 * sizeof(float));
        memcpy(x.data(), xd, 3 * sizeof(float));
        y.fill(1.0f);

        // y = 1.0 * A * x + 0.0 * y
        oil::math::gemv(1.0f, A, x, 0.0f, y);
        // y[0] = 1*2 + 2*3 + 3*4 = 20
        // y[1] = 4*2 + 5*3 + 6*4 = 47
        assert(std::abs(y.data<float>()[0] - 20.0f) < 1e-5f);
        assert(std::abs(y.data<float>()[1] - 47.0f) < 1e-5f);
    }

    // Test ReLU
    {
        auto x = oil::Tensor::zeros({5});
        float xd[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        memcpy(x.data(), xd, 5 * sizeof(float));
        auto y = oil::Tensor::zeros({5});
        oil::math::relu(x, y);
        assert(y.data<float>()[0] == 0.0f);
        assert(y.data<float>()[1] == 0.0f);
        assert(y.data<float>()[2] == 0.0f);
        assert(y.data<float>()[3] == 1.0f);
        assert(y.data<float>()[4] == 2.0f);
    }

    // Test softmax: sum to 1
    {
        auto x = oil::Tensor::zeros({3});
        float xd[] = {1.0f, 2.0f, 3.0f};
        memcpy(x.data(), xd, 3 * sizeof(float));
        auto y = oil::Tensor::zeros({3});
        oil::math::softmax(x, y, 0);

        float sum = 0.0f;
        for (int64_t i = 0; i < 3; i++)
            sum += y.data<float>()[i];
        assert(std::abs(sum - 1.0f) < 1e-5f);

        // All values positive
        for (int64_t i = 0; i < 3; i++)
            assert(y.data<float>()[i] > 0.0f);
    }

    // Test layer_norm: output has mean ~0 and variance ~1
    {
        int64_t D = 8;
        auto x = oil::Tensor::zeros({2, D});
        auto gamma = oil::Tensor::ones({D});
        auto beta = oil::Tensor::zeros({D});
        auto y = oil::Tensor::zeros({2, D});

        // Fill with increasing values
        for (int64_t i = 0; i < 2 * D; i++)
            x.data<float>()[i] = (float)(i + 1);

        oil::math::layer_norm(x, gamma, beta, 1e-5f, y);

        for (int64_t b = 0; b < 2; b++) {
            float mu = 0.0f;
            for (int64_t j = 0; j < D; j++)
                mu += y.data<float>()[b * D + j];
            mu /= D;
            assert(std::abs(mu) < 1e-5f);

            float var = 0.0f;
            for (int64_t j = 0; j < D; j++)
                var += (y.data<float>()[b * D + j] - mu) * (y.data<float>()[b * D + j] - mu);
            var /= D;
            assert(std::abs(var - 1.0f) < 1e-4f);
        }
    }

    // Test rms_norm
    {
        int64_t D = 8;
        auto x = oil::Tensor::zeros({D});
        auto gamma = oil::Tensor::ones({D});
        auto y = oil::Tensor::zeros({D});

        for (int64_t i = 0; i < D; i++)
            x.data<float>()[i] = (float)(i + 1);

        oil::math::rms_norm(x, gamma, 1e-5f, y);

        double ss = 0.0;
        for (int64_t i = 0; i < D; i++)
            ss += (double)x.data<float>()[i] * x.data<float>()[i];
        ss /= D;
        double inv = 1.0 / std::sqrt(ss + 1e-5); (void)inv;

        for (int64_t i = 0; i < D; i++)
            assert(std::abs(y.data<float>()[i] - (float)((double)x.data<float>()[i] * inv)) < 1e-5f);
    }

    // Test add/sub/mul/scale
    {
        auto a = oil::Tensor::zeros({4});
        auto b = oil::Tensor::zeros({4});
        auto c = oil::Tensor::zeros({4});
        a.fill(3.0f);
        b.fill(2.0f);

        oil::math::add(a, b, c);
        for (int64_t i = 0; i < 4; i++)
            assert(std::abs(c.data<float>()[i] - 5.0f) < 1e-6f);

        oil::math::sub(a, b, c);
        for (int64_t i = 0; i < 4; i++)
            assert(std::abs(c.data<float>()[i] - 1.0f) < 1e-6f);

        oil::math::mul(a, b, c);
        for (int64_t i = 0; i < 4; i++)
            assert(std::abs(c.data<float>()[i] - 6.0f) < 1e-6f);

        oil::math::scale(0.5f, a, c);
        for (int64_t i = 0; i < 4; i++)
            assert(std::abs(c.data<float>()[i] - 1.5f) < 1e-6f);
    }

    // Test mean / sum / max
    {
        auto x = oil::Tensor::arange(10);
        assert(std::abs(oil::math::sum(x) - 45.0f) < 1e-5f);
        assert(std::abs(oil::math::mean(x) - 4.5f) < 1e-5f);
        assert(std::abs(oil::math::max(x) - 9.0f) < 1e-5f);
    }

    // Test zeros_like / ones_like
    {
        auto x = oil::Tensor::arange(5);
        auto z = oil::math::zeros_like(x);
        assert(z.shape() == x.shape());
        for (int64_t i = 0; i < 5; i++)
            assert(z.data<float>()[i] == 0.0f);

        auto o = oil::math::ones_like(x);
        assert(o.shape() == x.shape());
        for (int64_t i = 0; i < 5; i++)
            assert(o.data<float>()[i] == 1.0f);
    }

    // Test gelu (smooth, non-zero for negatives)
    {
        auto x = oil::Tensor::zeros({4});
        float xd[] = {-2.0f, -1.0f, 0.0f, 1.0f};
        memcpy(x.data(), xd, 4 * sizeof(float));
        auto y = oil::Tensor::zeros({4});
        oil::math::gelu(x, y);
        assert(y.data<float>()[0] >= -0.1f);  // GELU(-2.0) ~ -0.045
        assert(y.data<float>()[1] < 0.0f);    // GELU(-1.0) is negative (~ -0.159)
        assert(y.data<float>()[1] > -0.3f);
        assert(std::abs(y.data<float>()[2]) < 1e-6f);
        assert(y.data<float>()[3] > 0.0f);
    }

    // Test sigmoid range
    {
        auto x = oil::Tensor::zeros({4});
        float xd[] = {-10.0f, 0.0f, 1.0f, 10.0f};
        memcpy(x.data(), xd, 4 * sizeof(float));
        auto y = oil::Tensor::zeros({4});
        oil::math::sigmoid(x, y);
        for (int64_t i = 0; i < 4; i++) {
            assert(y.data<float>()[i] > 0.0f);
            assert(y.data<float>()[i] < 1.0f);
        }
    }

    std::cout << "All math tests passed!" << std::endl;
    return 0;
}
