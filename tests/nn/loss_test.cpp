#include <gtest/gtest.h>

#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/loss.hpp"

TEST(LossTest, CrossEntropyKnownValue) {
  tiramisu::Tensor logits({2, 3});
  logits.at<float>({0, 0}) = 2.0f;
  logits.at<float>({0, 1}) = 1.0f;
  logits.at<float>({0, 2}) = 0.1f;
  logits.at<float>({1, 0}) = 0.1f;
  logits.at<float>({1, 1}) = 1.0f;
  logits.at<float>({1, 2}) = 2.0f;
  logits.set_requires_grad(true);

  tiramisu::Tensor targets({2});
  targets.at<float>({0}) = 0.0f;
  targets.at<float>({1}) = 2.0f;

  tiramisu::Tensor loss =
      tiramisu::nn::cross_entropy_loss(logits, targets);

  EXPECT_LT(loss.at<float>({0}), 0.5f);

  tiramisu::autograd::backward(loss);
  EXPECT_NE(logits.grad(), nullptr);
}
