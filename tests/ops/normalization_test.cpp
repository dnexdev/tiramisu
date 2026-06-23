#include <algorithm>

#include <gtest/gtest.h>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/normalization.hpp"

TEST(NormalizationTest, SoftmaxRowsSumToOne) {
  tiramisu::Tensor x({2, 4});
  const float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::copy(std::begin(vals), std::end(vals), x.data<float>());

  tiramisu::Tensor out = tiramisu::ops::softmax(x);
  const float* d = out.data<float>();

  EXPECT_NEAR(d[0] + d[1] + d[2] + d[3], 1.0f, 1e-5f);
  EXPECT_NEAR(d[4] + d[5] + d[6] + d[7], 1.0f, 1e-5f);
  EXPECT_NEAR(d[4], 0.25f, 1e-5f);
}

TEST(NormalizationTest, LayerNormOutputShapeAndMean) {
  tiramisu::Tensor x({3, 8});
  float* d = x.data<float>();
  for (int i = 0; i < 24; i++) {
    d[i] = static_cast<float>(i);
  }

  tiramisu::Tensor gamma({8}), beta({8});
  std::fill_n(gamma.data<float>(), 8, 1.0f);
  std::fill_n(beta.data<float>(), 8, 0.0f);

  tiramisu::Tensor out = tiramisu::ops::layernorm(x, gamma, beta);
  EXPECT_EQ(out.shape(), std::vector<int64_t>({3, 8}));

  const float* od = out.data<float>();
  for (int r = 0; r < 3; r++) {
    float mean = 0.0f;
    for (int i = 0; i < 8; i++) {
      mean += od[r * 8 + i];
    }
    mean /= 8.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);
  }
}
