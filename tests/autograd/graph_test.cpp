#include <gtest/gtest.h>

#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/tensor.hpp"

namespace tiramisu {

TEST(AutogradGraphTest, TensorUsedTwice) {
  Tensor x({1});
  x.at<float>({0}) = 3.0f;
  x.set_requires_grad(true);

  auto y = autograd::mul(x, x);
  auto z = autograd::add(y, y);

  autograd::backward(z);

  EXPECT_FLOAT_EQ(x.grad()->at<float>({0}), 12.0f);
}

}  // namespace tiramisu
