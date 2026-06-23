#include <gtest/gtest.h>

#include <memory>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/optim/sgd.hpp"

TEST(SGDTest, UpdatesParameters) {
  tiramisu::nn::Linear layer(2, 1);
  auto params = layer.parameters();

  tiramisu::optim::SGD opt(params, 0.01f);

  for (auto p : params) {
    p->set_requires_grad(true);
    auto grad = std::make_shared<tiramisu::Tensor>(p->shape(), p->dtype(),
                                                   p->device());
    std::fill_n(grad->data<float>(), grad->numel(), 1.0f);
    p->accumulate_grad(*grad);
  }

  float old_w = params[0]->data<float>()[0];
  opt.step();
  float new_w = params[0]->data<float>()[0];

  EXPECT_LT(new_w, old_w);
}

TEST(SGDTest, ZeroGradClearsGradients) {
  tiramisu::Tensor param({1});
  param.set_requires_grad(true);

  tiramisu::optim::SGD opt({&param});

  auto grad =
      std::make_shared<tiramisu::Tensor>(param.shape(), param.dtype(), param.device());
  grad->data<float>()[0] = 5.0f;
  param.accumulate_grad(*grad);

  opt.zero_grad();
  ASSERT_NE(param.grad(), nullptr);
  EXPECT_FLOAT_EQ(param.grad()->at<float>({0}), 0.0f);
}
