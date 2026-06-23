#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/matmul.hpp"

namespace {

void fill_sequence(tiramisu::Tensor& t, float start = 1.0f) {
  float* data = t.data<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    data[i] = start + static_cast<float>(i);
  }
}

void expect_tensors_near(const tiramisu::Tensor& actual,
                         const tiramisu::Tensor& expected, float tol) {
  ASSERT_EQ(actual.shape(), expected.shape());
  for (int64_t i = 0; i < actual.numel(); ++i) {
    EXPECT_NEAR(actual.data<float>()[i], expected.data<float>()[i], tol);
  }
}

}  // namespace

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

TEST(MatmulTest, BatchedMatmul) {
  tiramisu::Tensor a({2, 3, 4});
  tiramisu::Tensor b({2, 4, 2});
  fill_sequence(a);
  fill_sequence(b, 10.0f);

  tiramisu::Tensor c = tiramisu::ops::matmul(a, b);
  EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 3, 2}));

  for (int64_t batch = 0; batch < 2; ++batch) {
    tiramisu::Tensor a_batch = a.slice(0, batch, batch + 1);
    tiramisu::Tensor b_batch = b.slice(0, batch, batch + 1);
    tiramisu::Tensor expected = tiramisu::ops::matmul(a_batch, b_batch);
    tiramisu::Tensor actual = c.slice(0, batch, batch + 1);
    expect_tensors_near(actual, expected, 1e-4f);
  }
}

TEST(MatmulTest, BatchedBroadcast) {
  tiramisu::Tensor a({2, 3, 4});
  tiramisu::Tensor b({4, 2});
  fill_sequence(a);
  fill_sequence(b, 5.0f);

  tiramisu::Tensor c = tiramisu::ops::matmul(a, b);
  EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 3, 2}));

  for (int64_t batch = 0; batch < 2; ++batch) {
    tiramisu::Tensor a_batch = a.slice(0, batch, batch + 1);
    tiramisu::Tensor expected = tiramisu::ops::matmul(a_batch, b);
    tiramisu::Tensor actual = c.slice(0, batch, batch + 1);
    expect_tensors_near(actual, expected, 1e-4f);
  }
}

TEST(MatmulTest, MultipleBatchDimensions) {
  tiramisu::Tensor a({2, 2, 3, 4});
  tiramisu::Tensor b({2, 2, 4, 2});
  fill_sequence(a);
  fill_sequence(b, 20.0f);

  tiramisu::Tensor c = tiramisu::ops::matmul(a, b);
  EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 2, 3, 2}));

  tiramisu::Tensor a00 = a.slice(0, 0, 1).slice(0, 0, 1);
  tiramisu::Tensor b00 = b.slice(0, 0, 1).slice(0, 0, 1);
  tiramisu::Tensor expected00 = tiramisu::ops::matmul(a00, b00);
  tiramisu::Tensor actual00 = c.slice(0, 0, 1).slice(0, 0, 1);
  expect_tensors_near(actual00, expected00, 1e-4f);
}
