#include <cmath>

#include <gtest/gtest.h>

#include <memory>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/optim/adam.hpp"

TEST(AdamTest, UpdatesParameterOnStep) {
  tiramisu::Tensor param({1});
  param.at<float>({0}) = 1.0f;
  param.set_requires_grad(true);

  std::vector<tiramisu::Tensor*> params{&param};
  tiramisu::optim::Adam opt(params, 0.1f);

  auto grad =
      std::make_shared<tiramisu::Tensor>(param.shape(), param.dtype(), param.device());
  grad->data<float>()[0] = 1.0f;
  param.accumulate_grad(*grad);

  float before = param.at<float>({0});
  opt.step();
  float after = param.at<float>({0});

  EXPECT_LT(after, before);
}

TEST(AdamTest, FirstStepMatchesBiasCorrectedFormula) {
  tiramisu::Tensor param({1});
  param.at<float>({0}) = 0.0f;
  param.set_requires_grad(true);

  const float lr = 0.1f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;

  std::vector<tiramisu::Tensor*> params{&param};
  tiramisu::optim::Adam opt(params, lr, beta1, beta2, eps);

  auto grad =
      std::make_shared<tiramisu::Tensor>(param.shape(), param.dtype(), param.device());
  grad->data<float>()[0] = 1.0f;
  param.accumulate_grad(*grad);

  opt.step();

  float m_hat = 0.1f / (1.0f - beta1);
  float v_hat = 0.001f / (1.0f - beta2);
  float expected = -lr * m_hat / (std::sqrt(v_hat) + eps);

  EXPECT_NEAR(param.at<float>({0}), expected, 1e-5f);
}

TEST(AdamTest, ZeroGradClearsGradients) {
  tiramisu::Tensor param({1});
  param.set_requires_grad(true);

  std::vector<tiramisu::Tensor*> params{&param};
  tiramisu::optim::Adam opt(params);

  auto grad =
      std::make_shared<tiramisu::Tensor>(param.shape(), param.dtype(), param.device());
  grad->data<float>()[0] = 5.0f;
  param.accumulate_grad(*grad);

  opt.zero_grad();
  ASSERT_NE(param.grad(), nullptr);
  EXPECT_FLOAT_EQ(param.grad()->at<float>({0}), 0.0f);
}
