#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "tiramisu/nn/linear.hpp"

namespace tiramisu::serialize {

struct MNISTCheckpoint {
  static constexpr int64_t kInputDim = 784;
  static constexpr int64_t kHiddenDim = 128;
  static constexpr int64_t kOutputDim = 10;

  std::vector<float> w1;
  std::vector<float> b1;
  std::vector<float> w2;
  std::vector<float> b2;
};

void save_mnist_checkpoint(const std::string& path, const MNISTCheckpoint& ckpt);
void save_mnist_checkpoint(const std::string& path, nn::Linear& layer1,
                           nn::Linear& layer2);
MNISTCheckpoint load_mnist_checkpoint(const std::string& path);
MNISTCheckpoint load_mnist_checkpoint(std::span<const std::byte> data);
void load_mnist_checkpoint_into(const MNISTCheckpoint& ckpt, nn::Linear& layer1,
                                nn::Linear& layer2);

}  // namespace tiramisu::serialize
