#include <algorithm>

#include <gtest/gtest.h>

#include "tiramisu/autograd/gradcheck.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/linear.hpp"
#include "tiramisu/nn/layernorm.hpp"
#include "tiramisu/nn/multi_head_attention.hpp"

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

TEST(AutogradGradcheckTest, Gelu) {
  Tensor x({1});
  x.at<float>({0}) = 1.0f;
  EXPECT_TRUE(autograd::gradcheck(autograd::gelu, x, 1e-2, 1e-2));
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

TEST(AutogradGradcheckTest, BatchedMatmulWrtA) {
  Tensor B({1, 2, 2});
  B.at<float>({0, 0, 0}) = 1.0f;
  B.at<float>({0, 0, 1}) = 2.0f;
  B.at<float>({0, 1, 0}) = 3.0f;
  B.at<float>({0, 1, 1}) = 4.0f;

  Tensor A({1, 2, 2});
  A.at<float>({0, 0, 0}) = 5.0f;
  A.at<float>({0, 0, 1}) = 6.0f;
  A.at<float>({0, 1, 0}) = 7.0f;
  A.at<float>({0, 1, 1}) = 8.0f;

  auto f = [&B](const Tensor& t) {
    return autograd::sum(autograd::matmul(t, B));
  };
  EXPECT_TRUE(autograd::gradcheck(f, A, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, BatchedMatmulBroadcastWrtB) {
  Tensor W({2, 2});
  W.at<float>({0, 0}) = 1.0f;
  W.at<float>({0, 1}) = 2.0f;
  W.at<float>({1, 0}) = 3.0f;
  W.at<float>({1, 1}) = 4.0f;
  W.set_requires_grad(true);

  Tensor X({2, 2, 2});
  X.at<float>({0, 0, 0}) = 1.0f;
  X.at<float>({0, 0, 1}) = 2.0f;
  X.at<float>({0, 1, 0}) = 3.0f;
  X.at<float>({0, 1, 1}) = 4.0f;
  X.at<float>({1, 0, 0}) = 5.0f;
  X.at<float>({1, 0, 1}) = 6.0f;
  X.at<float>({1, 1, 0}) = 7.0f;
  X.at<float>({1, 1, 1}) = 8.0f;

  auto f = [&X](const Tensor& w) {
    return autograd::sum(autograd::matmul(X, w));
  };
  EXPECT_TRUE(autograd::gradcheck(f, W, 1e-2, 1e-2));
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

TEST(AutogradGradcheckTest, MhaOnLayerNorm) {
  nn::LayerNorm ln(8);
  nn::MultiHeadAttention mha(8, 2);
  Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.015f;
  }
  auto f = [&ln, &mha](const Tensor& t) {
    return autograd::sum(mha.forward(ln.forward(t)));
  };
  EXPECT_TRUE(autograd::gradcheck(f, x, 1e-3, 1e-1));
}

TEST(AutogradGradcheckTest, MhaResidualNoLn) {
  nn::MultiHeadAttention mha(8, 2);
  Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.015f;
  }
  auto f = [&mha](const Tensor& t) {
    return autograd::sum(autograd::add(t, mha.forward(t)));
  };
  EXPECT_TRUE(autograd::gradcheck(f, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, ReshapePermute) {
  Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.03f;
  }
  auto f = [](const Tensor& t) {
    Tensor y = autograd::permute(
        autograd::reshape(t, {1, 4, 2, 4}), {0, 2, 1, 3});
    return autograd::sum(y);
  };
  EXPECT_TRUE(autograd::gradcheck(f, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Linear3D) {
  nn::Linear layer(8, 8);
  Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.02f;
  }
  auto f = [&layer](const Tensor& t) {
    return autograd::sum(layer.forward(t));
  };
  EXPECT_TRUE(autograd::gradcheck(f, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, Linear3DSplitHeads) {
  nn::Linear layer(8, 8);
  Tensor x({1, 4, 8});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.02f;
  }
  auto f = [&layer](const Tensor& t) {
    Tensor q = layer.forward(t);
    Tensor y = autograd::permute(
        autograd::reshape(q, {1, 4, 2, 4}), {0, 2, 1, 3});
    return autograd::sum(y);
  };
  EXPECT_TRUE(autograd::gradcheck(f, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, MergeHeads) {
  Tensor x({1, 2, 4, 4});
  for (int64_t i = 0; i < x.numel(); ++i) {
    x.data<float>()[i] = static_cast<float>(i) * 0.02f;
  }
  auto f = [](const Tensor& t) {
    return autograd::sum(autograd::merge_heads(t, 8));
  };
  EXPECT_TRUE(autograd::gradcheck(f, x, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, MatmulTranspose) {
  Tensor a({1, 2, 4, 4});
  Tensor b({1, 2, 4, 4});
  for (int64_t i = 0; i < a.numel(); ++i) {
    a.data<float>()[i] = static_cast<float>(i) * 0.01f;
    b.data<float>()[i] = static_cast<float>(i) * 0.02f;
  }
  auto f = [&a](const Tensor& t) {
    return autograd::sum(
        autograd::matmul(a, autograd::transpose(t, -2, -1)));
  };
  EXPECT_TRUE(autograd::gradcheck(f, b, 1e-2, 1e-2));
}

TEST(AutogradGradcheckTest, AttentionScores) {
  Tensor q({1, 2, 4, 4});
  Tensor k({1, 2, 4, 4});
  for (int64_t i = 0; i < q.numel(); ++i) {
    q.data<float>()[i] = static_cast<float>(i) * 0.01f;
    k.data<float>()[i] = static_cast<float>(i) * 0.015f;
  }
  auto f = [&k](const Tensor& t) {
    Tensor scores =
        autograd::matmul(t, autograd::transpose(k, -2, -1));
    Tensor scale({1});
    scale.at<float>({0}) = 0.5f;
    scores = autograd::mul(scores, scale);
    Tensor mask({4, 4});
    for (int64_t i = 0; i < 4; ++i) {
      for (int64_t j = 0; j < 4; ++j) {
        mask.at<float>({i, j}) = (j > i) ? -1e9f : 0.0f;
      }
    }
    scores = autograd::add(scores, mask);
    return autograd::sum(autograd::softmax(scores));
  };
  EXPECT_TRUE(autograd::gradcheck(f, q, 1e-2, 1e-2));
}

}  // namespace tiramisu
