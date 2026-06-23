#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/elementwise.hpp"

TEST(ElementwiseTest, AddSameShape) {
  tiramisu::Tensor a({2, 2});
  tiramisu::Tensor b({2, 2});

  a.at<float>({0, 0}) = 1;
  a.at<float>({0, 1}) = 2;
  a.at<float>({1, 0}) = 3;
  a.at<float>({1, 1}) = 4;

  b.at<float>({0, 0}) = 5;
  b.at<float>({0, 1}) = 6;
  b.at<float>({1, 0}) = 7;
  b.at<float>({1, 1}) = 8;

  tiramisu::Tensor c = tiramisu::ops::add(a, b);

  EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 6);
  EXPECT_FLOAT_EQ(c.at<float>({0, 1}), 8);
  EXPECT_FLOAT_EQ(c.at<float>({1, 0}), 10);
  EXPECT_FLOAT_EQ(c.at<float>({1, 1}), 12);
}

TEST(ElementwiseTest, AddBroadcast) {
  tiramisu::Tensor matrix({2, 3});
  tiramisu::Tensor row_vector({1, 3});

  matrix.at<float>({0, 0}) = 1;
  matrix.at<float>({0, 1}) = 2;
  matrix.at<float>({0, 2}) = 3;
  matrix.at<float>({1, 0}) = 4;
  matrix.at<float>({1, 1}) = 5;
  matrix.at<float>({1, 2}) = 6;

  row_vector.at<float>({0, 0}) = 10;
  row_vector.at<float>({0, 1}) = 20;
  row_vector.at<float>({0, 2}) = 30;

  tiramisu::Tensor c = tiramisu::ops::add(matrix, row_vector);

  EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 3}));
  EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 11);
  EXPECT_FLOAT_EQ(c.at<float>({1, 2}), 36);
}

TEST(ElementwiseTest, MulBroadcast) {
  tiramisu::Tensor matrix({2, 3});
  tiramisu::Tensor row({1, 3});

  for (int i = 0; i < 6; i++) {
    matrix.data<float>()[i] = static_cast<float>(i + 1);
  }
  row.at<float>({0, 0}) = 2.0f;
  row.at<float>({0, 1}) = 2.0f;
  row.at<float>({0, 2}) = 2.0f;

  tiramisu::Tensor out = tiramisu::ops::mul(matrix, row);
  EXPECT_FLOAT_EQ(out.at<float>({0, 0}), 2.0f);
  EXPECT_FLOAT_EQ(out.at<float>({1, 2}), 12.0f);
}

TEST(ElementwiseTest, AddOnTransposedTensor) {
  tiramisu::Tensor a({2, 3});
  for (int i = 0; i < 6; i++) {
    a.data<float>()[i] = 1.0f;
  }

  tiramisu::Tensor b = a.transpose(0, 1);
  EXPECT_FALSE(b.is_contiguous());

  tiramisu::Tensor c = tiramisu::ops::add(b, b);
  EXPECT_EQ(c.shape(), std::vector<int64_t>({3, 2}));
  EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 2.0f);
}

TEST(ElementwiseTest, SubAndDiv) {
  tiramisu::Tensor a({2});
  a.at<float>({0}) = 10;
  a.at<float>({1}) = 20;
  tiramisu::Tensor b({2});
  b.at<float>({0}) = 2;
  b.at<float>({1}) = 5;

  auto t_sub = tiramisu::ops::sub(a, b);
  EXPECT_FLOAT_EQ(t_sub.at<float>({0}), 8);
  EXPECT_FLOAT_EQ(t_sub.at<float>({1}), 15);

  auto t_div = tiramisu::ops::div(a, b);
  EXPECT_FLOAT_EQ(t_div.at<float>({0}), 5);
  EXPECT_FLOAT_EQ(t_div.at<float>({1}), 4);
}

TEST(ElementwiseTest, UnaryOps) {
  tiramisu::Tensor a({3});
  a.at<float>({0}) = -5.0f;
  a.at<float>({1}) = 0.0f;
  a.at<float>({2}) = 3.0f;

  auto t_neg = tiramisu::ops::neg(a);
  EXPECT_FLOAT_EQ(t_neg.at<float>({0}), 5.0f);
  EXPECT_FLOAT_EQ(t_neg.at<float>({2}), -3.0f);

  auto t_relu = tiramisu::ops::relu(a);
  EXPECT_FLOAT_EQ(t_relu.at<float>({0}), 0.0f);
  EXPECT_FLOAT_EQ(t_relu.at<float>({2}), 3.0f);

  tiramisu::Tensor b({1});
  b.at<float>({0}) = 1.0f;
  auto t_exp = tiramisu::ops::exp(b);
  EXPECT_NEAR(t_exp.at<float>({0}), 2.71828f, 1e-4);

  auto t_log = tiramisu::ops::log(t_exp);
  EXPECT_NEAR(t_log.at<float>({0}), 1.0f, 1e-4);
}

TEST(ElementwiseTest, GeluKnownValues) {
  tiramisu::Tensor x({3});
  x.at<float>({0}) = 0.0f;
  x.at<float>({1}) = 1.0f;
  x.at<float>({2}) = -1.0f;

  tiramisu::Tensor y = tiramisu::ops::gelu(x);
  EXPECT_NEAR(y.at<float>({0}), 0.0f, 1e-5f);
  EXPECT_GT(y.at<float>({1}), 0.8f);
  EXPECT_LT(y.at<float>({1}), 1.0f);
  EXPECT_LT(y.at<float>({2}), 0.0f);
}
