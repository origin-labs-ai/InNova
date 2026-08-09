// test_ops.cpp — Math and autograd operations tests
#include "quant/math.h"
#include "quant/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "[Test] Running Ops test..." << std::endl;
    quant::Tensor a({2, 2}, quant::DType::F32);
    quant::Tensor b({2, 2}, quant::DType::F32);
    a.fill(2.0f);
    b.fill(3.0f);

    quant::Tensor c({2, 2}, quant::DType::F32);
    quant::math::add(a, b, c);

    const float* cd = c.data<float>();
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(cd[i] - 5.0f) < 1e-5f);
    }
    std::cout << "Ops test passed!" << std::endl;
    return 0;
}
