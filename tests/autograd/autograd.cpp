#include <gtest/gtest.h>
#include "tiramisu/autograd/ops.hpp"

TEST(AutogradTest, TensorUsedTwice) {
    tiramisu::Tensor x({1});
    x.at<float>({0}) = 3.0f;
    x.set_requires_grad(true);

    // y = x * x  -> dy/dx = 2x = 6
    auto y = tiramisu::autograd::mul(x, x);
    
    // z = y + y  -> dz/dy = 2, dz/dx = 4x = 12
    auto z = tiramisu::autograd::add(y, y);
    
    tiramisu::autograd::backward(z);
    
    EXPECT_FLOAT_EQ(x.grad()->at<float>({0}), 12.0f);
}