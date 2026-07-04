#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "tiramisu/nn/gpt.hpp"

namespace tiramisu::serialize {

// Kind tag for a v2 checkpoint record.
enum ParameterKind : uint32_t {
  kFp32 = 0,
  kInt8PerChannelSymmetric = 1,
};

struct ParameterEntry {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

// A single on-disk parameter, either fp32 or int8. Used only for writing
// quantized checkpoints; the public load path always dequantizes to fp32.
struct RawParameter {
  std::string name;
  std::vector<int64_t> shape;
  ParameterKind kind = kFp32;
  // Populated when kind == kFp32.
  std::vector<float> fp32_data;
  // Populated when kind == kInt8PerChannelSymmetric.
  int64_t channel_axis = 0;
  std::vector<float> scales;
  std::vector<int8_t> int8_data;
};

struct GPTCheckpoint {
  nn::GPTConfig config;
  int64_t step = 0;
  int64_t epoch = 0;
  std::vector<ParameterEntry> parameters;
};

struct RawGPTCheckpoint {
  nn::GPTConfig config;
  int64_t step = 0;
  int64_t epoch = 0;
  std::vector<RawParameter> parameters;
};

void save_gpt_checkpoint(const std::string& path, const GPTCheckpoint& ckpt);
GPTCheckpoint load_gpt_checkpoint(const std::string& path);
GPTCheckpoint load_gpt_checkpoint(std::span<const std::byte> data);

// Write a v2 checkpoint that may contain a mix of fp32 and int8 records.
// The load path dequantizes int8 records, so consumers get fp32 tensors.
void save_raw_gpt_checkpoint(const std::string& path,
                             const RawGPTCheckpoint& ckpt);

void save_gpt_model(const std::string& path, nn::GPT& model, int64_t step = 0,
                    int64_t epoch = 0);
void load_gpt_model(const std::string& path, nn::GPT& model, int64_t* step,
                    int64_t* epoch);
void load_gpt_checkpoint_into(const GPTCheckpoint& ckpt, nn::GPT& model);
nn::GPT create_gpt_from_checkpoint(std::span<const std::byte> data);

}  // namespace tiramisu::serialize
