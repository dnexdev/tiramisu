#include <gtest/gtest.h>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/tensor.hpp"

namespace tiramisu {

TEST(GradModeTest, NoGradGuardDisablesGraphConstruction) {
  Tensor x({2});
  x.at<float>({0}) = 1.0f;
  x.at<float>({1}) = 2.0f;
  x.set_requires_grad(true);

  Tensor y = x;
  {
    autograd::NoGradGuard guard;
    EXPECT_FALSE(autograd::grad_enabled());
    y = autograd::mul(x, x);
  }
  EXPECT_TRUE(autograd::grad_enabled());

  EXPECT_EQ(y.grad_fn(), nullptr);
  EXPECT_FALSE(y.requires_grad());
}

TEST(GradModeTest, NestedGuardsRestorePreviousState) {
  EXPECT_TRUE(autograd::grad_enabled());
  {
    autograd::NoGradGuard outer;
    EXPECT_FALSE(autograd::grad_enabled());
    {
      autograd::NoGradGuard inner;
      EXPECT_FALSE(autograd::grad_enabled());
    }
    EXPECT_FALSE(autograd::grad_enabled());
  }
  EXPECT_TRUE(autograd::grad_enabled());
}

}  // namespace tiramisu
