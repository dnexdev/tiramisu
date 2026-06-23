#include <gtest/gtest.h>

#include "tiramisu/autograd/gradcheck.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/nn/embedding.hpp"

TEST(EmbeddingTest, ForwardShape) {
  tiramisu::nn::Embedding emb(10, 4);
  tiramisu::Tensor ids({2, 3});
  ids.at<float>({0, 0}) = 1.0f;
  ids.at<float>({0, 1}) = 2.0f;
  ids.at<float>({0, 2}) = 3.0f;
  ids.at<float>({1, 0}) = 4.0f;
  ids.at<float>({1, 1}) = 5.0f;
  ids.at<float>({1, 2}) = 6.0f;

  tiramisu::Tensor out = emb.forward(ids);
  EXPECT_EQ(out.shape(), std::vector<int64_t>({2, 3, 4}));
}

TEST(EmbeddingTest, DuplicateIndicesAccumulateGrad) {
  tiramisu::nn::Embedding emb(4, 2);
  tiramisu::Tensor* weight = emb.parameters()[0];
  std::fill_n(weight->data<float>(), weight->numel(), 0.0f);

  tiramisu::Tensor ids({1, 2});
  ids.at<float>({0, 0}) = 1.0f;
  ids.at<float>({0, 1}) = 1.0f;

  tiramisu::Tensor out = emb.forward(ids);
  tiramisu::Tensor loss = tiramisu::autograd::sum(out);
  tiramisu::autograd::backward(loss);

  ASSERT_NE(weight->grad(), nullptr);
  EXPECT_FLOAT_EQ(weight->grad()->at<float>({1, 0}), 2.0f);
  EXPECT_FLOAT_EQ(weight->grad()->at<float>({1, 1}), 2.0f);
}

TEST(EmbeddingTest, Gradcheck) {
  tiramisu::Tensor weight({8, 3});
  for (int64_t i = 0; i < weight.numel(); ++i) {
    weight.data<float>()[i] = static_cast<float>(i) * 0.1f;
  }

  tiramisu::Tensor ids({1, 2});
  ids.at<float>({0, 0}) = 1.0f;
  ids.at<float>({0, 1}) = 3.0f;

  auto f = [&ids](const tiramisu::Tensor& w) {
    return tiramisu::autograd::sum(tiramisu::autograd::embedding(w, ids));
  };

  EXPECT_TRUE(tiramisu::autograd::gradcheck(f, weight, 1e-2, 1e-2));
}
