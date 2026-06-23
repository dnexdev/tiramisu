#include "tiramisu/nn/parameter.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "tiramisu/core/cuda_memory.hpp"

namespace tiramisu::nn {

Parameter::Parameter(const std::vector<int64_t>& shape, Device device)
    : Tensor(shape, DType::Float32, device) {
  set_requires_grad(true);

  std::vector<float> host(static_cast<size_t>(numel()));
  std::mt19937 gen(42);
  int64_t fan_in = (shape.size() >= 2) ? shape[1] : shape[0];
  float bound = std::sqrt(2.0f / static_cast<float>(fan_in));
  std::uniform_real_distribution<float> dist(-bound, bound);
  for (float& v : host) {
    v = dist(gen);
  }

  if (device == Device::CPU) {
    std::copy(host.begin(), host.end(), data<float>());
  } else {
    cuda_mem::copy_bytes(host.data(), data<float>(),
                         host.size() * sizeof(float), Device::CPU, device);
  }
}

}  // namespace tiramisu::nn
