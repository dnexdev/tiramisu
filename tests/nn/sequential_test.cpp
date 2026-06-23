#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/sequential.hpp"

TEST(SequentialTest, StackedLayers) {
  auto net = std::make_shared<tiramisu::nn::Sequential>(
      std::vector<std::shared_ptr<tiramisu::nn::Module>>{
          std::make_shared<tiramisu::nn::Linear>(3, 10),
          std::make_shared<tiramisu::nn::Linear>(10, 2)});

  tiramisu::Tensor x({1, 3});
  tiramisu::Tensor y = net->forward(x);
  EXPECT_EQ(y.shape(), std::vector<int64_t>({1, 2}));
}

TEST(SequentialTest, AggregatesParameters) {
  auto net = std::make_shared<tiramisu::nn::Sequential>(
      std::vector<std::shared_ptr<tiramisu::nn::Module>>{
          std::make_shared<tiramisu::nn::Linear>(3, 4),
          std::make_shared<tiramisu::nn::Linear>(4, 2)});

  auto params = net->parameters();
  EXPECT_EQ(params.size(), 4);
}
