#include <gtest/gtest.h>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/linear.hpp"

namespace tiramisu {

TEST(AutogradIntegrationTest, LinearBackwardToWeights) {
  nn::Linear layer(2, 1);

  Tensor* weight = layer.parameters()[0];
  Tensor* bias = layer.parameters()[1];
  std::fill_n(weight->data<float>(), weight->numel(), 1.0f);
  std::fill_n(bias->data<float>(), bias->numel(), 0.0f);

  Tensor x({1, 2});
  x.at<float>({0, 0}) = 1.0f;
  x.at<float>({0, 1}) = 2.0f;
  x.set_requires_grad(true);

  Tensor out = layer.forward(x);
  Tensor loss = autograd::sum(out);
  autograd::backward(loss);

  ASSERT_NE(weight->grad(), nullptr);
  EXPECT_FLOAT_EQ(weight->grad()->at<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(weight->grad()->at<float>({1, 0}), 2.0f);
  EXPECT_FLOAT_EQ(bias->grad()->at<float>({0}), 1.0f);
}

TEST(AutogradIntegrationTest, BiasBroadcastBackwardSumsOverBatch) {
  Tensor bias({2});
  bias.at<float>({0}) = 0.0f;
  bias.at<float>({1}) = 0.0f;
  bias.set_requires_grad(true);

  Tensor activations({3, 2});
  for (int i = 0; i < 6; i++) {
    activations.data<float>()[i] = 1.0f;
  }

  Tensor out = autograd::add(activations, bias);
  Tensor loss = autograd::sum(out);
  autograd::backward(loss);

  ASSERT_NE(bias.grad(), nullptr);
  EXPECT_EQ(bias.grad()->shape(), std::vector<int64_t>({2}));
  EXPECT_FLOAT_EQ(bias.grad()->at<float>({0}), 3.0f);
  EXPECT_FLOAT_EQ(bias.grad()->at<float>({1}), 3.0f);
}

TEST(AutogradIntegrationTest, InferenceSkipsGraphWithNoGradGuard) {
  nn::Linear layer(2, 1);
  Tensor x({1, 2});

  Tensor out = x;
  {
    autograd::NoGradGuard guard;
    out = layer.forward(x);
  }

  EXPECT_EQ(out.grad_fn(), nullptr);
}

TEST(AutogradIntegrationTest, BatchedMatmulBackward) {
  Tensor B({1, 3, 2});
  std::fill_n(B.data<float>(), B.numel(), 1.0f);
  Tensor A({1, 2, 3});
  std::fill_n(A.data<float>(), A.numel(), 1.0f);
  A.set_requires_grad(true);

  Tensor loss = autograd::sum(autograd::matmul(A, B));
  autograd::backward(loss);

  ASSERT_NE(A.grad(), nullptr);
  EXPECT_NEAR(A.grad()->at<float>({0, 0, 0}), 2.0f, 1e-4f);
}

TEST(AutogradIntegrationTest, BatchedMatmulBroadcastBackward) {
  Tensor W({3, 2});
  std::fill_n(W.data<float>(), W.numel(), 1.0f);
  W.set_requires_grad(true);

  Tensor X({2, 2, 3});
  std::fill_n(X.data<float>(), X.numel(), 1.0f);

  Tensor loss = autograd::sum(autograd::matmul(X, W));
  autograd::backward(loss);

  ASSERT_NE(W.grad(), nullptr);
  EXPECT_NEAR(W.grad()->at<float>({0, 0}), 4.0f, 1e-4f);
}

}  // namespace tiramisu
