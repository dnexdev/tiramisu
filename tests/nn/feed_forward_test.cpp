#include <gtest/gtest.h>

#include "tiramisu/autograd/gradcheck.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/nn/feed_forward.hpp"

TEST(FeedForwardTest, PreservesShape) {
  tiramisu::nn::FeedForward ffn(8);
  tiramisu::Tensor x({2, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.01f;
  }

  tiramisu::Tensor y = ffn.forward(x);
  EXPECT_EQ(y.shape(), std::vector<int64_t>({2, 4, 8}));
}

TEST(FeedForwardTest, Gradcheck) {
  tiramisu::nn::FeedForward ffn(8);
  tiramisu::Tensor x({1, 3, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.03f;
  }

  auto f = [&ffn](const tiramisu::Tensor& t) {
    return tiramisu::autograd::sum(ffn.forward(t));
  };

  EXPECT_TRUE(tiramisu::autograd::gradcheck(f, x, 1e-2, 1e-2));
}
