#include <gtest/gtest.h>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/autograd/gradcheck.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/nn/gpt.hpp"
#include "tiramisu/nn/loss.hpp"

namespace {

tiramisu::nn::GPTConfig tiny_config() {
  return tiramisu::nn::GPTConfig{
      .vocab_size = 32,
      .d_model = 16,
      .num_heads = 2,
      .num_layers = 1,
      .max_seq_len = 8,
  };
}

}  // namespace

TEST(GPTTest, OutputShape) {
  tiramisu::nn::GPT model(tiny_config());
  tiramisu::Tensor ids({2, 4});
  ids.at<float>({0, 0}) = 1.0f;
  ids.at<float>({0, 1}) = 2.0f;
  ids.at<float>({0, 2}) = 3.0f;
  ids.at<float>({0, 3}) = 4.0f;
  ids.at<float>({1, 0}) = 5.0f;
  ids.at<float>({1, 1}) = 6.0f;
  ids.at<float>({1, 2}) = 7.0f;
  ids.at<float>({1, 3}) = 8.0f;

  tiramisu::Tensor logits = model.forward(ids);
  EXPECT_EQ(logits.shape(), std::vector<int64_t>({2, 4, 32}));
}

TEST(GPTTest, ParameterCountGrowsWithLayers) {
  auto cfg1 = tiny_config();
  cfg1.num_layers = 1;
  auto cfg2 = tiny_config();
  cfg2.num_layers = 2;

  tiramisu::nn::GPT model1(cfg1);
  tiramisu::nn::GPT model2(cfg2);

  EXPECT_LT(model1.parameters().size(), model2.parameters().size());
}

TEST(GPTTest, EndToEndGradcheck) {
  tiramisu::nn::GPT model(tiny_config());
  tiramisu::Tensor ids({1, 4});
  for (int64_t i = 0; i < 4; ++i) {
    ids.at<float>({0, i}) = static_cast<float>(i + 1);
  }

  tiramisu::Tensor loss = tiramisu::autograd::sum(model.forward(ids));
  tiramisu::autograd::backward(loss);

  for (tiramisu::Tensor* p : model.parameters()) {
    ASSERT_NE(p->grad(), nullptr);
    float grad_sq = 0.0f;
    for (int64_t i = 0; i < p->grad()->numel(); ++i) {
      const float g = p->grad()->data<float>()[i];
      grad_sq += g * g;
    }
    EXPECT_GT(grad_sq, 0.0f);
  }
}

TEST(GPTTest, LossIntegration) {
  tiramisu::nn::GPT model(tiny_config());
  tiramisu::Tensor ids({2, 4});
  for (int64_t i = 0; i < ids.numel(); ++i) {
    ids.data<float>()[i] = static_cast<float>(i % 16);
  }

  tiramisu::Tensor targets({2, 4});
  for (int64_t i = 0; i < targets.numel(); ++i) {
    targets.data<float>()[i] = static_cast<float>((i + 1) % 16);
  }

  tiramisu::Tensor logits = model.forward(ids);
  const int64_t batch = logits.shape()[0];
  const int64_t seq = logits.shape()[1];
  const int64_t vocab = logits.shape()[2];

  tiramisu::Tensor flat_logits =
      tiramisu::autograd::reshape(logits, {batch * seq, vocab});
  tiramisu::Tensor flat_targets = targets.reshape({batch * seq});

  tiramisu::Tensor loss =
      tiramisu::nn::cross_entropy_loss(flat_logits, flat_targets);
  tiramisu::autograd::backward(loss);

  const std::vector<tiramisu::Tensor*> params = model.parameters();
  tiramisu::Tensor* tok_weight = params[0];
  tiramisu::Tensor* lm_weight = params[params.size() - 2];

  ASSERT_NE(tok_weight->grad(), nullptr);
  ASSERT_NE(lm_weight->grad(), nullptr);

  float tok_grad_sq = 0.0f;
  for (int64_t i = 0; i < tok_weight->grad()->numel(); ++i) {
    const float g = tok_weight->grad()->data<float>()[i];
    tok_grad_sq += g * g;
  }
  float lm_grad_sq = 0.0f;
  for (int64_t i = 0; i < lm_weight->grad()->numel(); ++i) {
    const float g = lm_weight->grad()->data<float>()[i];
    lm_grad_sq += g * g;
  }

  EXPECT_GT(tok_grad_sq, 0.0f);
  EXPECT_GT(lm_grad_sq, 0.0f);
}
