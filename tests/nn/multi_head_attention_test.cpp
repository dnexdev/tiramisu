#include <gtest/gtest.h>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/autograd/gradcheck.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/nn/multi_head_attention.hpp"

TEST(MultiHeadAttentionTest, OutputShape) {
  tiramisu::nn::MultiHeadAttention mha(8, 2);
  tiramisu::Tensor x({2, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.01f;
  }

  tiramisu::Tensor y = mha.forward(x);
  EXPECT_EQ(y.shape(), std::vector<int64_t>({2, 4, 8}));
}

TEST(MultiHeadAttentionTest, CausalMaskBlocksFuture) {
  tiramisu::nn::MultiHeadAttention mha(8, 2);
  tiramisu::Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.05f;
  }

  tiramisu::Tensor out1 = [&]() {
    tiramisu::autograd::NoGradGuard guard;
    return mha.forward(x);
  }();

  x.at<float>({0, 3, 0}) += 100.0f;
  tiramisu::Tensor out2 = [&]() {
    tiramisu::autograd::NoGradGuard guard;
    return mha.forward(x);
  }();

  for (int64_t s = 0; s < 3; ++s) {
    for (int64_t d = 0; d < 8; ++d) {
      EXPECT_FLOAT_EQ(out1.at<float>({0, s, d}), out2.at<float>({0, s, d}));
    }
  }
}

TEST(MultiHeadAttentionTest, Gradcheck) {
  tiramisu::nn::MultiHeadAttention mha(8, 2);
  tiramisu::Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.02f;
  }

  auto f = [&mha](const tiramisu::Tensor& t) {
    return tiramisu::autograd::sum(mha.forward(t));
  };

  EXPECT_TRUE(tiramisu::autograd::gradcheck(f, x, 1e-2, 1e-2));
}
