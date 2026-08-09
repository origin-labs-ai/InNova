#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/test.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    TEST_SUITE("Tensor Tests");
    // Test Shape
    {
        quant::Shape s1(2, 3);
        TEST_CHECK(s1.numel() == 6, "Shape(2,3).numel() == 6");
        TEST_CHECK(s1.rank == 2, "Shape(2,3).rank == 2");
        TEST_CHECK(s1.dims[0] == 2, "Shape(2,3).dims[0] == 2");
        TEST_CHECK(s1.dims[1] == 3, "Shape(2,3).dims[1] == 3");

        quant::Shape s2(4, 5, 6);
        TEST_CHECK(s2.rank == 3, "Shape(4,5,6).rank == 3");
        TEST_CHECK(s2.numel() == 120, "Shape(4,5,6).numel() == 120");

        quant::Shape s3;
        TEST_CHECK(s3.rank == 0, "default Shape.rank == 0");
        TEST_CHECK(s3.numel() == 1, "default Shape.numel() == 1");

        TEST_CHECK(s1 == quant::Shape(2, 3), "Shape equality");
        TEST_CHECK(!(s1 == s2), "Shape inequality");
        TEST_CHECK(s1 != s2, "Shape != operator");
    }

    // Test Tensor creation
    {
        quant::Tensor t(quant::Shape(2, 3), quant::DType::F32);
        TEST_CHECK(t.numel() == 6, "Tensor(2x3 F32).numel() == 6");
        TEST_CHECK(t.dtype() == quant::DType::F32, "F32 dtype");
        TEST_CHECK(t.shape().rank == 2, "shape().rank == 2");
        TEST_CHECK(t.shape().dims[0] == 2, "shape().dims[0] == 2");
        TEST_CHECK(t.shape().dims[1] == 3, "shape().dims[1] == 3");
        bool zero_init = true;
        for (int64_t i = 0; i < 6; i++)
            if (t.data<float>()[i] != 0.0f) zero_init = false;
        TEST_CHECK(zero_init, "zero initialized");
    }

    // Test fill
    {
        quant::Tensor t(quant::Shape(2, 2), quant::DType::F32);
        t.fill(1.0f);
        bool filled = true;
        for (int64_t i = 0; i < 4; i++)
            if (t.data<float>()[i] != 1.0f) filled = false;
        TEST_CHECK(filled, "fill(1.0)");
    }

    // Test init with arange
    {
        quant::Tensor t = quant::Tensor::arange(5);
        bool match = true;
        for (int64_t i = 0; i < 5; i++)
            if (t.data<float>()[i] != (float)i) match = false;
        TEST_CHECK(match, "arange(5)");
    }

    // Test fill and access
    {
        quant::Tensor t(quant::Shape(2, 3), quant::DType::F32);
        for (int64_t i = 0; i < 6; i++) t.data<float>()[i] = (float)i;
        TEST_CHECK(t.at({0, 0}) == 0.0f, "at({0,0}) == 0");
        TEST_CHECK(t.at({0, 1}) == 1.0f, "at({0,1}) == 1");
        TEST_CHECK(t.at({1, 2}) == 5.0f, "at({1,2}) == 5");
    }

    // Test view
    {
        quant::Tensor t(quant::Shape(2, 3), quant::DType::F32);
        for (int64_t i = 0; i < 6; i++) t.data<float>()[i] = (float)i;
        auto v = t.view({2, 3});
        TEST_CHECK(v.shape().rank == 2, "view rank == 2");
        TEST_CHECK(v.shape().dims[0] == 2, "view dims[0] == 2");
        TEST_CHECK(v.shape().dims[1] == 3, "view dims[1] == 3");
        TEST_CHECK(v.at({0, 0}) == 0.0f, "view at({0,0}) == 0");
        TEST_CHECK(v.at({1, 2}) == 5.0f, "view at({1,2}) == 5");
        t.data<float>()[4] = 42.0f;
        TEST_CHECK(v.at({1, 1}) == 42.0f, "view reflects mutation");
    }

    // Test reshape
    {
        quant::Tensor t(quant::Shape(2, 6), quant::DType::F32);
        for (int64_t i = 0; i < 12; i++) t.data<float>()[i] = (float)i;
        auto r = t.reshape({3, 4});
        TEST_CHECK(r.shape().dims[0] == 3, "reshape dims[0] == 3");
        TEST_CHECK(r.shape().dims[1] == 4, "reshape dims[1] == 4");
        TEST_CHECK(r.numel() == 12, "reshape numel == 12");
    }

    // Test slice on 1D tensor
    {
        quant::Tensor t = quant::Tensor::arange(5);
        auto s = t.slice(0, 1, 4);
        TEST_CHECK(s.numel() == 3, "slice 1D [1,4) numel == 3");
        bool ok = s.data<float>()[0] == 1.0f && s.data<float>()[1] == 2.0f && s.data<float>()[2] == 3.0f;
        TEST_CHECK(ok, "slice 1D values 1,2,3");
    }

    // Test transpose
    {
        quant::Tensor t = quant::Tensor::arange(6);
        t = t.reshape({2, 3});
        auto tp = t.transpose(0, 1);
        TEST_CHECK(tp.shape().dims[0] == 3, "transpose dims[0] == 3");
        TEST_CHECK(tp.shape().dims[1] == 2, "transpose dims[1] == 2");
        TEST_CHECK(tp.at({0, 0}) == t.at({0, 0}), "transpose at({0,0})");
        TEST_CHECK(tp.at({1, 0}) == t.at({0, 1}), "transpose at({1,0}) == at({0,1})");
        TEST_CHECK(tp.at({2, 1}) == t.at({1, 2}), "transpose at({2,1}) == at({1,2})");
    }

    // Test clone
    {
        quant::Tensor t(quant::Shape(2, 3), quant::DType::F32);
        for (int64_t i = 0; i < 6; i++) t.data<float>()[i] = (float)i;
        auto c = t.clone();
        TEST_CHECK(c.numel() == t.numel(), "clone numel");
        bool match = true;
        for (int64_t i = 0; i < 6; i++)
            if (c.data<float>()[i] != t.data<float>()[i]) match = false;
        TEST_CHECK(match, "clone values match");
        t.data<float>()[0] = 100.0f;
        TEST_CHECK(c.data<float>()[0] == 0.0f, "clone independent");
    }

    // Test copy constructor and assignment
    {
        quant::Tensor t(quant::Shape(2, 3), quant::DType::F32);
        for (int64_t i = 0; i < 6; i++) t.data<float>()[i] = (float)i;
        quant::Tensor t2 = t;
        TEST_CHECK(t2.numel() == t.numel(), "copy numel");
        TEST_CHECK(t2.shape() == t.shape(), "copy shape");
        TEST_CHECK(t2.dtype() == t.dtype(), "copy dtype");
        bool match = true;
        for (int64_t i = 0; i < 6; i++)
            if (std::abs(t2.data<float>()[i] - t.data<float>()[i]) > 1e-6f) match = false;
        TEST_CHECK(match, "copy values");
        t2 = quant::Tensor(quant::Shape(1, 1), quant::DType::F32);
        TEST_CHECK(t2.numel() == 1, "re-assign numel");
    }

    // Test zeros/ones static
    {
        auto z = quant::Tensor::zeros({2, 3});
        TEST_CHECK(z.numel() == 6, "zeros numel");
        bool zero = true;
        for (int64_t i = 0; i < 6; i++)
            if (z.data<float>()[i] != 0.0f) zero = false;
        TEST_CHECK(zero, "zeros all zero");

        auto o = quant::Tensor::ones({2, 2});
        bool one = true;
        for (int64_t i = 0; i < 4; i++)
            if (o.data<float>()[i] != 1.0f) one = false;
        TEST_CHECK(one, "ones all one");
    }

    // Test copy_from
    {
        quant::Tensor a(quant::Shape(3), quant::DType::F32);
        for (int64_t i = 0; i < 3; i++) a.data<float>()[i] = (float)i;
        quant::Tensor b(quant::Shape(3), quant::DType::F32);
        b.copy_from(a);
        bool match = true;
        for (int64_t i = 0; i < 3; i++)
            if (b.data<float>()[i] != (float)i) match = false;
        TEST_CHECK(match, "copy_from values");
    }

    // Test to_string
    {
        quant::Tensor t(quant::Shape(2, 3), quant::DType::F32);
        auto s = t.to_string();
        TEST_CHECK(!s.empty(), "to_string() non-empty");
        TEST_CHECK(s.find("Tensor") != std::string::npos, "to_string() contains Tensor");
    }

    // Test to_dtype
    {
        quant::Tensor t(quant::Shape(2, 2), quant::DType::F32);
        t.fill(3.14f);
        auto t2 = t.to_dtype(quant::DType::F16);
        TEST_CHECK(t2.numel() == 4, "to_dtype numel preserved");
        TEST_CHECK(t2.dtype() == quant::DType::F16, "to_dtype F16");
    }

    return TEST_REPORT() > 0 ? 1 : 0;
}
