#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/matmul.hpp"

TEST(MatmulTest, TwoByThreeTimesThreeByTwo) {
  tiramisu::Tensor a({2, 3});
  a.at<float>({0, 0}) = 1;
  a.at<float>({0, 1}) = 2;
  a.at<float>({0, 2}) = 3;
  a.at<float>({1, 0}) = 4;
  a.at<float>({1, 1}) = 5;
  a.at<float>({1, 2}) = 6;

  tiramisu::Tensor b({3, 2});
  b.at<float>({0, 0}) = 7;
  b.at<float>({0, 1}) = 8;
  b.at<float>({1, 0}) = 9;
  b.at<float>({1, 1}) = 10;
  b.at<float>({2, 0}) = 11;
  b.at<float>({2, 1}) = 12;

  tiramisu::Tensor c = tiramisu::ops::matmul(a, b);

  EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 2}));
  EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 58);
  EXPECT_FLOAT_EQ(c.at<float>({0, 1}), 64);
  EXPECT_FLOAT_EQ(c.at<float>({1, 0}), 139);
  EXPECT_FLOAT_EQ(c.at<float>({1, 1}), 154);
}
