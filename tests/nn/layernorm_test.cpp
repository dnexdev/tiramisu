#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/layernorm.hpp"

TEST(LayerNormTest, ForwardPreservesShape) {
  tiramisu::nn::LayerNorm norm(4);

  tiramisu::Tensor x({2, 4});
  for (int i = 0; i < 8; i++) {
    x.data<float>()[i] = static_cast<float>(i + 1);
  }

  tiramisu::Tensor y = norm.forward(x);
  EXPECT_EQ(y.shape(), std::vector<int64_t>({2, 4}));
}

TEST(LayerNormTest, ExposesGammaAndBeta) {
  tiramisu::nn::LayerNorm norm(8);
  auto params = norm.parameters();
  EXPECT_EQ(params.size(), 2);
  EXPECT_EQ(params[0]->shape(), std::vector<int64_t>({8}));
  EXPECT_EQ(params[1]->shape(), std::vector<int64_t>({8}));
}

TEST(LayerNormTest, InitializesGammaToOneAndBetaToZero) {
  tiramisu::nn::LayerNorm norm(3);
  const float* gamma = norm.parameters()[0]->data<float>();
  const float* beta = norm.parameters()[1]->data<float>();

  for (int i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(gamma[i], 1.0f);
    EXPECT_FLOAT_EQ(beta[i], 0.0f);
  }
}
