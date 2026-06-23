#include <cmath>

#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/optim/adamw.hpp"

TEST(AdamWTest, AppliesWeightDecay) {
  tiramisu::Tensor param({1});
  param.at<float>({0}) = 1.0f;
  param.set_requires_grad(true);

  std::vector<tiramisu::Tensor*> params{&param};
  tiramisu::optim::AdamW opt(params, 0.1f, 0.9f, 0.999f, 1e-8f, 0.5f);

  auto grad = std::make_shared<tiramisu::Tensor>(param.shape());
  grad->data<float>()[0] = 0.0f;
  param.accumulate_grad(*grad);

  opt.step();
  EXPECT_LT(param.at<float>({0}), 1.0f);
}

TEST(AdamWTest, ZeroGradClearsGradients) {
  tiramisu::Tensor param({1});
  param.set_requires_grad(true);
  std::vector<tiramisu::Tensor*> params{&param};
  tiramisu::optim::AdamW opt(params);

  auto grad = std::make_shared<tiramisu::Tensor>(param.shape());
  grad->data<float>()[0] = 2.0f;
  param.accumulate_grad(*grad);

  opt.zero_grad();
  ASSERT_NE(param.grad(), nullptr);
  EXPECT_FLOAT_EQ(param.grad()->at<float>({0}), 0.0f);
}
