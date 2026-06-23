#include <gtest/gtest.h>

#include "tiramisu/autograd/gradcheck.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/nn/transformer_block.hpp"

TEST(TransformerBlockTest, PreservesShape) {
  tiramisu::nn::TransformerBlock block(8, 2);
  tiramisu::Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.01f;
  }

  tiramisu::Tensor y = block.forward(x);
  EXPECT_EQ(y.shape(), std::vector<int64_t>({1, 4, 8}));
}

TEST(TransformerBlockTest, GradFlowsToParameters) {
  tiramisu::nn::TransformerBlock block(8, 2);
  tiramisu::Tensor x({1, 3, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.02f;
  }
  x.set_requires_grad(true);

  tiramisu::Tensor loss = tiramisu::autograd::sum(block.forward(x));
  tiramisu::autograd::backward(loss);

  int64_t params_with_grad = 0;
  for (tiramisu::Tensor* p : block.parameters()) {
    if (p->grad() != nullptr) {
      params_with_grad++;
    }
  }
  EXPECT_EQ(params_with_grad, static_cast<int64_t>(block.parameters().size()));
}

TEST(TransformerBlockTest, Gradcheck) {
  tiramisu::nn::TransformerBlock block(8, 2);
  tiramisu::Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.015f;
  }

  auto f = [&block](const tiramisu::Tensor& t) {
    return tiramisu::autograd::sum(block.forward(t));
  };

  EXPECT_TRUE(tiramisu::autograd::gradcheck(f, x, 1e-3, 1e-1));
}
