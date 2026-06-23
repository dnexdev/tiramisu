#include "tiramisu/nn/parameter.hpp"

#include <random>

namespace tiramisu::nn {

Parameter::Parameter(const std::vector<int64_t>& shape) : Tensor(shape) {
  this->set_requires_grad(true);

  std::mt19937 gen(42);
  int64_t fan_in = (shape.size() >= 2) ? shape[1] : shape[0];
  float bound = std::sqrt(2.0f / static_cast<float>(fan_in));
  std::uniform_real_distribution<float> dist(-bound, bound);

  float* data_ptr = this->data<float>();
  int64_t num_elements_ = this->numel();

  for (int64_t i = 0; i < num_elements_; i++) {
    data_ptr[i] = dist(gen);
  }
}

}  // namespace tiramisu::nn