#include "tiramisu/nn/feed_forward.hpp"

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

FeedForward::FeedForward(int64_t d_model)
    : fc1_(d_model, 4 * d_model), fc2_(4 * d_model, d_model) {}

Tensor FeedForward::forward(const Tensor& x) {
  return fc2_.forward(tiramisu::autograd::gelu(fc1_.forward(x)));
}

std::vector<Tensor*> FeedForward::parameters() {
  std::vector<Tensor*> params = fc1_.parameters();
  auto fc2_params = fc2_.parameters();
  params.insert(params.end(), fc2_params.begin(), fc2_params.end());
  return params;
}

}  // namespace tiramisu::nn
