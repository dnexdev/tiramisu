#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"

TEST(TensorAutogradTest, AccumulateGradSumsInPlace) {
  tiramisu::Tensor t({2});
  t.set_requires_grad(true);

  tiramisu::Tensor g1({2});
  g1.data<float>()[0] = 1.0f;
  g1.data<float>()[1] = 2.0f;

  tiramisu::Tensor g2({2});
  g2.data<float>()[0] = 3.0f;
  g2.data<float>()[1] = 4.0f;

  t.accumulate_grad(g1);
  t.accumulate_grad(g2);

  ASSERT_NE(t.grad(), nullptr);
  EXPECT_FLOAT_EQ(t.grad()->at<float>({0}), 4.0f);
  EXPECT_FLOAT_EQ(t.grad()->at<float>({1}), 6.0f);
}

TEST(TensorAutogradTest, CopySharesAutogradState) {
  tiramisu::Tensor a({1});
  a.set_requires_grad(true);
  a.at<float>({0}) = 2.0f;

  tiramisu::Tensor b = a;
  b.at<float>({0}) = 5.0f;

  EXPECT_FLOAT_EQ(a.at<float>({0}), 5.0f);
  EXPECT_TRUE(b.requires_grad());
}

TEST(TensorAutogradTest, DataThrowsOnDtypeMismatch) {
  tiramisu::Tensor t({1}, tiramisu::DType::Float32);
  EXPECT_NO_THROW(t.data<float>());
  EXPECT_THROW(t.data<int32_t>(), std::runtime_error);
}
