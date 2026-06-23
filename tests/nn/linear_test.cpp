#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/linear.hpp"

TEST(LinearTest, ForwardPass) {
  tiramisu::nn::Linear layer(3, 2);
  tiramisu::Tensor x({1, 3});
  x.at<float>({0, 0}) = 1.0f;
  x.at<float>({0, 1}) = 2.0f;
  x.at<float>({0, 2}) = 3.0f;

  tiramisu::Tensor y = layer.forward(x);
  EXPECT_EQ(y.shape(), std::vector<int64_t>({1, 2}));
}

TEST(LinearTest, ExposesWeightAndBias) {
  tiramisu::nn::Linear layer(2, 1);
  auto params = layer.parameters();
  EXPECT_EQ(params.size(), 2);
}
