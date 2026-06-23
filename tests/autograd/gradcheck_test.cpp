#include <algorithm>

#include <gtest/gtest.h>

#include "tiramisu/autograd/gradcheck.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/tensor.hpp"

namespace tiramisu {

TEST(AutogradGradcheckTest, Add) {
  Tensor a({1}), b({1});
  a.at<float>({0}) = 3.0f;
  b.at<float>({0}) = 4.0f;
  auto f = [&b](const Tensor& t) { return autograd::add(t, b); };
  EXPECT_TRUE(autograd::gradcheck(f, a, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Mul) {
  Tensor a({1}), b({1});
  a.at<float>({0}) = 3.0f;
  b.at<float>({0}) = 5.0f;
  auto f = [&b](const Tensor& t) {
    return autograd::sum(autograd::mul(t, b));
  };
  EXPECT_TRUE(autograd::gradcheck(f, a, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Sub) {
  Tensor a({1}), b({1});
  a.at<float>({0}) = 5.0f;
  b.at<float>({0}) = 2.0f;
  auto f = [&b](const Tensor& t) { return autograd::sub(t, b); };
  EXPECT_TRUE(autograd::gradcheck(f, a, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Div) {
  Tensor a({1}), b({1});
  a.at<float>({0}) = 6.0f;
  b.at<float>({0}) = 2.0f;
  auto f = [&b](const Tensor& t) { return autograd::div(t, b); };
  EXPECT_TRUE(autograd::gradcheck(f, a, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Neg) {
  Tensor x({1});
  x.at<float>({0}) = 4.0f;
  EXPECT_TRUE(autograd::gradcheck(autograd::neg, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Exp) {
  Tensor x({1});
  x.at<float>({0}) = 1.0f;
  EXPECT_TRUE(autograd::gradcheck(autograd::exp, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Log) {
  Tensor x({1});
  x.at<float>({0}) = 2.0f;
  EXPECT_TRUE(autograd::gradcheck(autograd::log, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, ReluPositiveBranch) {
  Tensor x({1});
  x.at<float>({0}) = 2.0f;
  EXPECT_TRUE(autograd::gradcheck(autograd::relu, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Sum) {
  Tensor x({4});
  for (int i = 0; i < 4; i++) {
    x.at<float>({i}) = static_cast<float>(i + 1);
  }
  EXPECT_TRUE(autograd::gradcheck(autograd::sum, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Mean) {
  Tensor x({4});
  for (int i = 0; i < 4; i++) {
    x.at<float>({i}) = static_cast<float>(i + 1);
  }
  EXPECT_TRUE(autograd::gradcheck(autograd::mean, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, MatmulWrtA) {
  Tensor B({2, 2});
  B.at<float>({0, 0}) = 1;
  B.at<float>({0, 1}) = 2;
  B.at<float>({1, 0}) = 3;
  B.at<float>({1, 1}) = 4;

  Tensor A({2, 2});
  A.at<float>({0, 0}) = 5;
  A.at<float>({0, 1}) = 6;
  A.at<float>({1, 0}) = 7;
  A.at<float>({1, 1}) = 8;

  auto f = [&B](const Tensor& t) {
    return autograd::sum(autograd::matmul(t, B));
  };
  EXPECT_TRUE(autograd::gradcheck(f, A, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, MatmulWrtB) {
  Tensor A({2, 2});
  A.at<float>({0, 0}) = 1;
  A.at<float>({0, 1}) = 2;
  A.at<float>({1, 0}) = 3;
  A.at<float>({1, 1}) = 4;

  Tensor B({2, 2});
  B.at<float>({0, 0}) = 5;
  B.at<float>({0, 1}) = 6;
  B.at<float>({1, 0}) = 7;
  B.at<float>({1, 1}) = 8;

  auto f = [&A](const Tensor& t) {
    return autograd::sum(autograd::matmul(A, t));
  };
  EXPECT_TRUE(autograd::gradcheck(f, B, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Softmax) {
  Tensor x({1, 4});
  x.at<float>({0, 0}) = 1.0f;
  x.at<float>({0, 1}) = 2.0f;
  x.at<float>({0, 2}) = 3.0f;
  x.at<float>({0, 3}) = 4.0f;

  auto f = [](const Tensor& t) {
    return autograd::sum(autograd::softmax(t));
  };
  EXPECT_TRUE(autograd::gradcheck(f, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, LayerNorm) {
  Tensor x({2, 4});
  const float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f};
  std::copy(std::begin(vals), std::end(vals), x.data<float>());

  Tensor gamma({4}), beta({4});
  std::fill_n(gamma.data<float>(), 4, 1.0f);
  std::fill_n(beta.data<float>(), 4, 0.0f);
  gamma.set_requires_grad(true);

  auto f = [&gamma, &beta](const Tensor& t) {
    return autograd::sum(autograd::layernorm(t, gamma, beta));
  };
  EXPECT_TRUE(autograd::gradcheck(f, x, 1e-2, 1e-2));
}

}  // namespace tiramisu
