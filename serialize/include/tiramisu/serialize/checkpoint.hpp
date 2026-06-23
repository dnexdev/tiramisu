#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tiramisu/nn/gpt.hpp"

namespace tiramisu::serialize {

struct ParameterEntry {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

struct GPTCheckpoint {
  nn::GPTConfig config;
  int64_t step = 0;
  int64_t epoch = 0;
  std::vector<ParameterEntry> parameters;
};

void save_gpt_checkpoint(const std::string& path, const GPTCheckpoint& ckpt);
GPTCheckpoint load_gpt_checkpoint(const std::string& path);

void save_gpt_model(const std::string& path, nn::GPT& model, int64_t step = 0,
                    int64_t epoch = 0);
void load_gpt_model(const std::string& path, nn::GPT& model, int64_t* step,
                    int64_t* epoch);

}  // namespace tiramisu::serialize
