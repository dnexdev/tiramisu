#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/reduce.hpp"

TEST(ReduceTest, SumAndMean) {
  tiramisu::Tensor a({4});
  a.at<float>({0}) = 2;
  a.at<float>({1}) = 4;
  a.at<float>({2}) = 6;
  a.at<float>({3}) = 8;

  tiramisu::Tensor s = tiramisu::ops::sum(a);
  tiramisu::Tensor m = tiramisu::ops::mean(a);

  EXPECT_FLOAT_EQ(s.at<float>({0}), 20);
  EXPECT_FLOAT_EQ(m.at<float>({0}), 5);
}
