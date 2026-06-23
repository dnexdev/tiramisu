#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "tiramisu/nn/gpt.hpp"
#include "tiramisu/serialize/checkpoint.hpp"

namespace {

tiramisu::nn::GPTConfig tiny_config() {
  return tiramisu::nn::GPTConfig{
      .vocab_size = 16,
      .d_model = 8,
      .num_heads = 2,
      .num_layers = 1,
      .max_seq_len = 4,
      .tie_weights = true,
  };
}

}  // namespace

TEST(CheckpointTest, RoundTrip) {
  const std::string path = "checkpoint_test.bin";
  std::filesystem::remove(path);

  tiramisu::nn::GPT model(tiny_config());
  const float before = model.parameters()[0]->at<float>({0, 0});

  tiramisu::serialize::save_gpt_model(path, model, 42, 3);

  tiramisu::nn::GPT loaded(tiny_config());
  int64_t step = 0;
  int64_t epoch = 0;
  tiramisu::serialize::load_gpt_model(path, loaded, &step, &epoch);

  EXPECT_EQ(step, 42);
  EXPECT_EQ(epoch, 3);
  EXPECT_FLOAT_EQ(loaded.parameters()[0]->at<float>({0, 0}), before);

  std::filesystem::remove(path);
}
