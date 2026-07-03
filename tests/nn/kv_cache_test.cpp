#include <gtest/gtest.h>
#include <random>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/nn/gpt.hpp"
#include "tiramisu/nn/kv_cache.hpp"

namespace {

tiramisu::Tensor make_ids(const std::vector<int64_t>& ids) {
  tiramisu::Tensor t({1, static_cast<int64_t>(ids.size())});
  for (size_t i = 0; i < ids.size(); ++i) {
    t.data<float>()[i] = static_cast<float>(ids[i]);
  }
  return t;
}

int64_t argmax_logits(const tiramisu::Tensor& logits) {
  const int64_t vocab = logits.shape().back();
  const float* data = logits.data<float>();
  int64_t best = 0;
  for (int64_t i = 1; i < vocab; ++i) {
    if (data[i] > data[best]) {
      best = i;
    }
  }
  return best;
}

tiramisu::nn::GPTConfig tiny_config() {
  return tiramisu::nn::GPTConfig{
      .vocab_size = 32,
      .d_model = 16,
      .num_heads = 2,
      .num_layers = 2,
      .max_seq_len = 8,
  };
}

std::vector<int64_t> generate_tokens_naive(tiramisu::nn::GPT& model,
                                           const std::vector<int64_t>& prompt,
                                           int64_t max_new_tokens) {
  tiramisu::autograd::NoGradGuard guard;
  std::vector<int64_t> context = prompt;

  for (int64_t t = 0; t < max_new_tokens; ++t) {
    tiramisu::Tensor ids = make_ids(context);
    tiramisu::Tensor logits = model.forward(ids);
    const int64_t last = logits.shape()[1] - 1;
    const int64_t vocab = logits.shape().back();
    tiramisu::Tensor flat = logits.reshape({logits.shape()[1], vocab});
    tiramisu::Tensor row = flat.slice(0, last, last + 1).reshape({vocab});
    context.push_back(argmax_logits(row));
    if (context.size() > static_cast<size_t>(model.config().max_seq_len)) {
      context.erase(context.begin());
    }
  }

  return std::vector<int64_t>(context.begin() + static_cast<long>(prompt.size()),
                              context.end());
}

std::vector<int64_t> generate_tokens_cached(tiramisu::nn::GPT& model,
                                            const std::vector<int64_t>& prompt,
                                            int64_t max_new_tokens) {
  tiramisu::autograd::NoGradGuard guard;
  std::vector<int64_t> context = prompt;

  tiramisu::nn::GPTKVCache cache;
  tiramisu::Tensor logits = model.prefill(make_ids(context), cache);
  for (int64_t t = 0; t < max_new_tokens; ++t) {
    const int64_t next =
        argmax_logits(logits.reshape({model.config().vocab_size}).contiguous());
    context.push_back(next);
    if (context.size() > static_cast<size_t>(model.config().max_seq_len)) {
      context.erase(context.begin());
      cache.reset();
      cache.layers.resize(static_cast<size_t>(model.config().num_layers));
      logits = model.prefill(make_ids(context), cache);
    } else {
      logits = model.decode_step(next, cache);
    }
  }

  return std::vector<int64_t>(context.begin() + static_cast<long>(prompt.size()),
                              context.end());
}

}  // namespace

TEST(KVCacheTest, DecodeStepMatchesForwardOneStep) {
  tiramisu::nn::GPT model(tiny_config());
  const std::vector<int64_t> prompt = {1, 2, 3};

  tiramisu::nn::GPTKVCache cache;
  tiramisu::Tensor prefill_logits = model.prefill(make_ids(prompt), cache);
  const int64_t t1 = argmax_logits(
      prefill_logits.reshape({model.config().vocab_size}).contiguous());

  tiramisu::Tensor decoded =
      model.decode_step(t1, cache).reshape({model.config().vocab_size}).contiguous();

  std::vector<int64_t> ctx = prompt;
  ctx.push_back(t1);
  tiramisu::Tensor full = model.forward(make_ids(ctx));
  const int64_t vocab = full.shape().back();
  tiramisu::Tensor flat = full.reshape({full.shape()[1], vocab});
  tiramisu::Tensor full_last =
      flat.slice(0, full.shape()[1] - 1, full.shape()[1]).reshape({vocab});

  for (int64_t i = 0; i < vocab; ++i) {
    EXPECT_NEAR(decoded.data<float>()[i], full_last.data<float>()[i], 1e-4f);
  }
}

TEST(KVCacheTest, PrefillMatchesForwardLastLogits) {
  tiramisu::nn::GPT model(tiny_config());
  const std::vector<int64_t> ids = {1, 3, 5, 2};

  tiramisu::nn::GPTKVCache cache;
  tiramisu::Tensor cached = model.prefill(make_ids(ids), cache);
  tiramisu::Tensor full = model.forward(make_ids(ids));
  const int64_t last = full.shape()[1] - 1;
  const int64_t vocab = full.shape().back();
  tiramisu::Tensor flat = full.reshape({full.shape()[1], vocab});
  tiramisu::Tensor full_last = flat.slice(0, last, last + 1).reshape({vocab});

  for (int64_t i = 0; i < full_last.numel(); ++i) {
    EXPECT_NEAR(cached.data<float>()[i], full_last.data<float>()[i], 1e-4f);
  }
}

TEST(KVCacheTest, CachedGenerationMatchesNaiveGreedy) {
  tiramisu::nn::GPT model(tiny_config());
  const std::vector<int64_t> prompt = {1, 2, 3};

  EXPECT_EQ(generate_tokens_cached(model, prompt, 12),
            generate_tokens_naive(model, prompt, 12));
}

TEST(KVCacheTest, CachedGenerationMatchesNaiveWithSlidingWindow) {
  tiramisu::nn::GPT model(tiny_config());
  const std::vector<int64_t> prompt = {1, 2, 3, 4, 5, 6, 7};

  EXPECT_EQ(generate_tokens_cached(model, prompt, 20),
            generate_tokens_naive(model, prompt, 20));
}
