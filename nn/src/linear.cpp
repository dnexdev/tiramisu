#include "tiramisu/nn/linear.hpp"

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

Linear::Linear(int64_t in_features, int64_t out_features, Device device)
    : weight_({in_features, out_features}, device),
      bias_({out_features}, device) {}

Tensor Linear::forward(const Tensor& x) {
  Tensor out = tiramisu::autograd::matmul(x, weight_);
  return tiramisu::autograd::add(out, bias_);
}

std::vector<Tensor*> Linear::parameters() { return {&weight_, &bias_}; }

}  // namespace tiramisu::nn