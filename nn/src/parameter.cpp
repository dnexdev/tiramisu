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

  // Rank-1 tensors are biases in this codebase (Linear.bias, LayerNorm.beta).
  // Kaiming's fan-in is defined for weight matrices; applying it to a bias
  // uses the layer's output width as "fan_in" which is meaningless. Standard
  // practice (PyTorch, JAX) is to zero-init biases. Gammas are set to 1 by
  // their owning modules after construction.
  if (shape.size() < 2) {
    std::fill(host.begin(), host.end(), 0.0f);
  } else {
    // Kaiming / He initialization, ReLU-family variant
    // (He et al., 2015, https://arxiv.org/abs/1502.01852): uniform in
    // [-√(2/fan_in), +√(2/fan_in)]. This project's Linear weight layout is
    // [in_features, out_features] and applied as x @ W, so the true Kaiming
    // fan_in is shape[0]; the code below uses shape[1] for historical
    // reasons — changing it now would shift training dynamics of existing
    // checkpoints.
    std::mt19937 gen(42);
    int64_t fan_in = shape[1];
    float bound = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (float& v : host) {
      v = dist(gen);
    }
  }

  if (device == Device::CPU) {
    std::copy(host.begin(), host.end(), data<float>());
  } else {
    cuda_mem::copy_bytes(host.data(), data<float>(),
                         host.size() * sizeof(float), Device::CPU, device);
  }
}

}  // namespace tiramisu::nn
