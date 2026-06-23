#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"

TEST(TensorSliceTest, ExtractsLeadingDimension) {
  tiramisu::Tensor t({4, 3});
  for (int64_t i = 0; i < 12; i++) {
    t.data<float>()[i] = static_cast<float>(i);
  }

  tiramisu::Tensor batch = t.slice(0, 1, 3);
  EXPECT_EQ(batch.shape(), std::vector<int64_t>({2, 3}));
  EXPECT_FLOAT_EQ(batch.at<float>({0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(batch.at<float>({0, 2}), 5.0f);
  EXPECT_FLOAT_EQ(batch.at<float>({1, 0}), 6.0f);
}

TEST(TensorSliceTest, SharesStorageWithParent) {
  tiramisu::Tensor t({3, 2});
  t.data<float>()[0] = 1.0f;

  tiramisu::Tensor view = t.slice(0, 0, 2);
  view.data<float>()[0] = 99.0f;

  EXPECT_FLOAT_EQ(t.data<float>()[0], 99.0f);
}

TEST(TensorSliceTest, RejectsNonZeroDimension) {
  tiramisu::Tensor t({3, 2});
  EXPECT_THROW(t.slice(1, 0, 2), std::runtime_error);
}
