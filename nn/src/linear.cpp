#include "tiramisu/nn/linear.hpp"

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

Linear::Linear(int64_t in_features, int64_t out_features)
    : weight({in_features, out_features}), bias({out_features}) {}

Tensor Linear::forward(const Tensor& x) {
  // x shape : (batch_size, in_features)
  // weight shape : (out_features, in_features)
  // we compute : x @ weight ^ T + bias

  // Tensor w_t = weight.transpose(0, 1);
  Tensor out = tiramisu::autograd::matmul(x, weight);
  return tiramisu::autograd::add(out, bias);
}

std::vector<Tensor*> Linear::parameters() { return {&weight, &bias}; }

}  // namespace tiramisu::nn