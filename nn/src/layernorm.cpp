#include "tiramisu/nn/layernorm.hpp"

#include <algorithm>

#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/cuda_memory.hpp"

namespace tiramisu::nn {

LayerNorm::LayerNorm(int64_t features, float eps, Device device)
    : gamma_({features}, device), beta_({features}, device), eps_(eps) {
  if (device == Device::CPU) {
    std::fill_n(gamma_.data<float>(), features, 1.0f);
    std::fill_n(beta_.data<float>(), features, 0.0f);
  } else {
    cuda_mem::fill_f32(gamma_.data<float>(), 1.0f, features, device);
    cuda_mem::fill_f32(beta_.data<float>(), 0.0f, features, device);
  }
}

Tensor LayerNorm::forward(const Tensor& x) {
  return tiramisu::autograd::layernorm(x, gamma_, beta_, eps_);
}

std::vector<Tensor*> LayerNorm::parameters() {
  return {&gamma_, &beta_};
}

}  // namespace tiramisu::nn
