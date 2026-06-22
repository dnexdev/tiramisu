#include "tiramisu/core/tensor.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "tiramisu/core/device.hpp"
#include "tiramisu/core/dtype.hpp"

TEST(TensorUtils, ContiguousStrides) {
  // 1D
  auto strides_1d = tiramisu::contiguous_strides({10});
  EXPECT_EQ(strides_1d, std::vector<int64_t>({1}));

  // 2D Matrix (3x4)
  auto strides_2d = tiramisu::contiguous_strides({3, 4});
  EXPECT_EQ(strides_2d, std::vector<int64_t>({4, 1}));

  // 3D Volume (2x3x4)
  auto strides_3d = tiramisu::contiguous_strides({2, 3, 4});
  EXPECT_EQ(strides_3d, std::vector<int64_t>({12, 4, 1}));
}

TEST(TensorTest, FreshAllocation) {
  tiramisu::Tensor t({2, 3, 4}, tiramisu::DType::Float32,
                     tiramisu::Device::CPU);

  EXPECT_EQ(t.shape(), std::vector<int64_t>({2, 3, 4}));
  EXPECT_EQ(t.strides(), std::vector<int64_t>({12, 4, 1}));
  EXPECT_EQ(t.numel(), 24);
  EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, DataAccessAndAt) {
  tiramisu::Tensor t({2, 2}, tiramisu::DType::Float32, tiramisu::Device::CPU);

  float* raw_data = t.data<float>();

  // Fill using raw pointer
  raw_data[0] = 1.0f;  // [0, 0]
  raw_data[1] = 2.0f;  // [0, 1]
  raw_data[2] = 3.0f;  // [1, 0]
  raw_data[3] = 4.0f;  // [1, 1]

  // Read back using .at()
  EXPECT_FLOAT_EQ(t.at<float>({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at<float>({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(t.at<float>({1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(t.at<float>({1, 1}), 4.0f);

  // Write using .at()
  t.at<float>({1, 1}) = 99.0f;
  EXPECT_FLOAT_EQ(raw_data[3], 99.0f);

  // Bounds checking
  EXPECT_THROW(t.at<float>({2, 0}), std::out_of_range);
}

TEST(TensorTest, View) {
  tiramisu::Tensor t({2, 6});  // 12 elements
  EXPECT_TRUE(t.is_contiguous());

  // Valid view
  tiramisu::Tensor v = t.view({3, 4});
  EXPECT_EQ(v.shape(), std::vector<int64_t>({3, 4}));
  EXPECT_EQ(v.strides(), std::vector<int64_t>({4, 1}));
  EXPECT_TRUE(v.is_contiguous());

  // Invalid view (wrong element count)
  EXPECT_THROW(t.view({3, 5}), std::invalid_argument);
}

TEST(TensorTest, TransposeAndPermute) {
  tiramisu::Tensor t({2, 3, 4});  // Strides: {12, 4, 1}

  // 1. Transpose: Swap dim 0 and dim 1 -> Shape {3, 2, 4}
  tiramisu::Tensor trans = t.transpose(0, 1);
  EXPECT_EQ(trans.shape(), std::vector<int64_t>({3, 2, 4}));
  EXPECT_EQ(trans.strides(),
            std::vector<int64_t>({4, 12, 1}));  // Note the swapped strides
  EXPECT_FALSE(trans.is_contiguous());

  // 2. Permute: Reverse all dims -> Shape {4, 3, 2}
  tiramisu::Tensor perm = t.permute({2, 1, 0});
  EXPECT_EQ(perm.shape(), std::vector<int64_t>({4, 3, 2}));
  EXPECT_EQ(perm.strides(), std::vector<int64_t>({1, 4, 12}));
  EXPECT_FALSE(perm.is_contiguous());
}

TEST(TensorTest, MakeContiguous) {
  tiramisu::Tensor t({2, 3});

  // Write sequential data
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      t.at<float>({i, j}) = static_cast<float>(i * 3 + j);
    }
  }
  // Layout: [0, 1, 2, 3, 4, 5]

  // Transpose creates a non-contiguous view: Shape {3, 2}, Strides {1, 3}
  tiramisu::Tensor view = t.transpose(0, 1);
  EXPECT_FALSE(view.is_contiguous());

  // Call contiguous to trigger a deep copy packing
  tiramisu::Tensor contig = view.contiguous();
  EXPECT_TRUE(contig.is_contiguous());
  EXPECT_EQ(contig.shape(), std::vector<int64_t>({3, 2}));
  EXPECT_EQ(contig.strides(),
            std::vector<int64_t>({2, 1}));  // Brand new canonical strides

  // Ensure data was copied into the correct geometric position
  // The mathematical transpose of:
  // [0, 1, 2]
  // [3, 4, 5]
  // is:
  // [0, 3]
  // [1, 4]
  // [2, 5]
  EXPECT_FLOAT_EQ(contig.at<float>({0, 0}), 0.0f);
  EXPECT_FLOAT_EQ(contig.at<float>({0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(contig.at<float>({1, 0}), 1.0f);
  EXPECT_FLOAT_EQ(contig.at<float>({1, 1}), 4.0f);
  EXPECT_FLOAT_EQ(contig.at<float>({2, 0}), 2.0f);
  EXPECT_FLOAT_EQ(contig.at<float>({2, 1}), 5.0f);

  // Verify flat memory matches the repacked geometry
  float* contig_data = contig.data<float>();
  EXPECT_FLOAT_EQ(contig_data[1], 3.0f);
}

TEST(TensorTest, ScalarNumelMatchesEmptyShapeConvention) {
  tiramisu::Tensor t({1});
  EXPECT_EQ(t.numel(), 1);

  tiramisu::Tensor scalar = t.view({});
  EXPECT_EQ(scalar.shape().size(), 0u);
  EXPECT_TRUE(scalar.is_contiguous());
  EXPECT_EQ(scalar.numel(), 1);
}