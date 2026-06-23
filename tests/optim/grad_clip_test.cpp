#include <cmath>

#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/optim/grad_clip.hpp"

TEST(GradClipTest, ScalesLargeGradients) {
  tiramisu::Tensor p({2});
  p.set_requires_grad(true);
  auto grad = std::make_shared<tiramisu::Tensor>(p.shape());
  grad->data<float>()[0] = 3.0f;
  grad->data<float>()[1] = 4.0f;
  p.accumulate_grad(*grad);

  std::vector<tiramisu::Tensor*> params{&p};
  const float norm = tiramisu::optim::clip_grad_norm(params, 2.5f);
  EXPECT_NEAR(norm, 5.0f, 1e-5f);

  const float clipped_norm = std::sqrt(p.grad()->at<float>({0}) * p.grad()->at<float>({0}) +
                                       p.grad()->at<float>({1}) * p.grad()->at<float>({1}));
  EXPECT_NEAR(clipped_norm, 2.5f, 1e-4f);
}

TEST(GradClipTest, LeavesSmallGradientsUntouched) {
  tiramisu::Tensor p({1});
  p.set_requires_grad(true);
  auto grad = std::make_shared<tiramisu::Tensor>(p.shape());
  grad->data<float>()[0] = 0.5f;
  p.accumulate_grad(*grad);

  std::vector<tiramisu::Tensor*> params{&p};
  tiramisu::optim::clip_grad_norm(params, 1.0f);
  EXPECT_FLOAT_EQ(p.grad()->at<float>({0}), 0.5f);
}
