#include <gtest/gtest.h>

#include "tiramisu/autograd/gradcheck.hpp"
#include "tiramisu/autograd/ops.hpp"

namespace tiramisu {
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

TEST(AutogradOpsTest, SubGradient) {
  Tensor a({1}), b({1});
  a.at<float>({0}) = 5.0f;
  b.at<float>({0}) = 2.0f;
  auto f = [&b](const Tensor& t) { return autograd::sub(t, b); };
  EXPECT_TRUE(autograd::gradcheck(f, a, 1e-2, 1e-2));
}

TEST(AutogradOpsTest, DivGradient) {
  Tensor a({1}), b({1});
  a.at<float>({0}) = 6.0f;
  b.at<float>({0}) = 2.0f;
  auto f = [&b](const Tensor& t) { return autograd::div(t, b); };
  EXPECT_TRUE(autograd::gradcheck(f, a, 1e-2, 1e-2));
}

TEST(AutogradOpsTest, NegGradient) {
  Tensor x({1});
  x.at<float>({0}) = 4.0f;
  EXPECT_TRUE(autograd::gradcheck(autograd::neg, x, 1e-2, 1e-2));
}

TEST(AutogradOpsTest, ExpGradient) {
  Tensor x({1});
  x.at<float>({0}) = 1.0f;
  EXPECT_TRUE(autograd::gradcheck(autograd::exp, x, 1e-2, 1e-2));
}

TEST(AutogradOpsTest, LogGradient) {
  Tensor x({1});
  x.at<float>({0}) = 2.0f;  // away from 0, where log blows up
  EXPECT_TRUE(autograd::gradcheck(autograd::log, x, 1e-2, 1e-2));
}

TEST(AutogradOpsTest, ReluGradientPositiveBranch) {
  Tensor x({1});
  x.at<float>({0}) = 2.0f;  // away from the kink at 0 -- gradcheck
                            // near 0 would be unreliable either way
  EXPECT_TRUE(autograd::gradcheck(autograd::relu, x, 1e-2, 1e-2));
}

TEST(AutogradOpsTest, SumGradient) {
  Tensor x({4});  // multi-element on purpose -- this is what actually
                  // exercises the broadcast-back-to-shape logic
  for (int i = 0; i < 4; i++) x.at<float>({i}) = static_cast<float>(i + 1);
  EXPECT_TRUE(autograd::gradcheck(autograd::sum, x, 1e-2, 1e-2));
}

TEST(AutogradOpsTest, MeanGradient) {
  Tensor x({4});
  for (int i = 0; i < 4; i++) x.at<float>({i}) = static_cast<float>(i + 1);
  EXPECT_TRUE(autograd::gradcheck(autograd::mean, x, 1e-2, 1e-2));
}

TEST(AutogradOpsTest, MatmulGradientWrtA) {
  // implement sum/mean before this one -- matmul's output isn't
  // scalar, so scalarizing through sum() is what makes gradcheck
  // applicable here at all.
  Tensor B({2, 2});
  B.at<float>({0, 0}) = 1;
  B.at<float>({0, 1}) = 2;
  B.at<float>({1, 0}) = 3;
  B.at<float>({1, 1}) = 4;

  Tensor A({2, 2});
  A.at<float>({0, 0}) = 5;
  A.at<float>({0, 1}) = 6;
  A.at<float>({1, 0}) = 7;
  A.at<float>({1, 1}) = 8;

  auto f = [&B](const Tensor& t) {
    return autograd::sum(autograd::matmul(t, B));
  };
  EXPECT_TRUE(autograd::gradcheck(f, A, 1e-2, 1e-2));
}
}  // namespace tiramisu