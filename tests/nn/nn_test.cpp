#include <gtest/gtest.h>
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/sequential.hpp"
#include "tiramisu/optim/sgd.hpp"
#include "tiramisu/nn/loss.hpp"
#include "tiramisu/autograd/ops.hpp"

TEST(LinearTest, ForwardPass) {
  tiramisu::nn::Linear layer(3, 2);
  tiramisu::Tensor x({1, 3});  // batch=1, in_features=3
  x.at<float>({0, 0}) = 1.0f;
  x.at<float>({0, 1}) = 2.0f;
  x.at<float>({0, 2}) = 3.0f;
  
  tiramisu::Tensor y = layer.forward(x);
  EXPECT_EQ(y.shape(), std::vector<int64_t>({1, 2}));
}

TEST(LinearTest, ParameterGradients) {
  tiramisu::nn::Linear layer(2, 1);
  auto params = layer.parameters();
  EXPECT_EQ(params.size(), 2);  // weight + bias
}

TEST(SequentialTest, StackedLayers) {
  auto net = std::make_shared<tiramisu::nn::Sequential>(
    std::vector<std::shared_ptr<tiramisu::nn::Module>>{
      std::make_shared<tiramisu::nn::Linear>(3, 10),
      std::make_shared<tiramisu::nn::Linear>(10, 2)
    }
  );
  
  tiramisu::Tensor x({1, 3});
  tiramisu::Tensor y = net->forward(x);
  EXPECT_EQ(y.shape(), std::vector<int64_t>({1, 2}));
}

TEST(SGDTest, UpdatesParameters) {
  tiramisu::nn::Linear layer(2, 1);
  auto params = layer.parameters();
  
  tiramisu::optim::SGD opt(params, 0.01f);
  
  // Set gradients manually for this test
  for (auto p : params) {
    p->set_requires_grad(true);
    auto grad = std::make_shared<tiramisu::Tensor>(p->shape(), p->dtype(), p->device());
    std::fill_n(grad->data<float>(), grad->numel(), 1.0f);
    p->accumulate_grad(*grad);
  }
  
  float old_w = params[0]->data<float>()[0];
  opt.step();
  float new_w = params[0]->data<float>()[0];
  
  // Weight should decrease by (lr * grad) = 0.01
  EXPECT_LT(new_w, old_w);
}

TEST(LossTest, CrossEntropyKnownValue) {
  // batch=2, classes=3
  tiramisu::Tensor logits({2, 3});
  // row 0: class 0 has highest logit
  logits.at<float>({0, 0}) = 2.0f;
  logits.at<float>({0, 1}) = 1.0f;
  logits.at<float>({0, 2}) = 0.1f;
  // row 1: class 2 has highest logit
  logits.at<float>({1, 0}) = 0.1f;
  logits.at<float>({1, 1}) = 1.0f;
  logits.at<float>({1, 2}) = 2.0f;
  logits.set_requires_grad(true);

  // targets: sample 0 -> class 0 (correct), sample 1 -> class 2 (correct)
  tiramisu::Tensor targets({2});
  targets.at<float>({0}) = 0.0f;
  targets.at<float>({1}) = 2.0f;

  tiramisu::Tensor loss = tiramisu::nn::cross_entropy_loss(logits, targets);

  // Loss should be low (both predictions correct, high confidence)
  EXPECT_LT(loss.at<float>({0}), 0.5f);

  // Gradient should flow back to logits
  tiramisu::autograd::backward(loss);
  EXPECT_NE(logits.grad(), nullptr);
}