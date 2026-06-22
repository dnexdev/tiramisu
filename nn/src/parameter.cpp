#include "tiramisu/nn/parameter.hpp"
#include <random>

namespace tiramisu::nn {

Parameter::Parameter(const std::vector<int64_t>& shape) : Tensor(shape) {
  this->set_requires_grad(true);

  // random initialization uniformly in [-1, 1]
  // seed is fixed for reproducibility, will be changed to random_device in prod
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  float* data_ptr = this->data<float>();
  int64_t num_elements_ = this->numel();

  for (int64_t i = 0; i < num_elements_; i++) {
    data_ptr[i] = dist(gen);
  }
}

}