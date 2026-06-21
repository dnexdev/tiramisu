#pragma once

#include <functional>
#include <vector>
#include "tiramisu/core/tensor.hpp"

namespace tiramisu {

struct Node {
  std::vector<Tensor> inputs;

  std::function<std::vector<Tensor>(const Tensor& grad_output)> backward_fn;
};

}