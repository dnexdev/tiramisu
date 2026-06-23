#include "tiramisu/nn/layernorm.hpp"

#include <algorithm>

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

LayerNorm::LayerNorm(int64_t features, float eps)
  : gamma_({features}), beta_({features}), eps_(eps) {
  
  std::fill_n(gamma_.data<float>(), features, 1.0f);
  std::fill_n(beta_.data<float>(), features, 0.0f);
}

Tensor LayerNorm::forward(const Tensor& x) {
  return tiramisu::autograd::layernorm(x, gamma_, beta_, eps_);
}

std::vector<Tensor*> LayerNorm::parameters() {
  return {&gamma_, &beta_};
}

}