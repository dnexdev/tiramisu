#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::nn {

class Parameter : public Tensor {
 public:
  Parameter(const std::vector<int64_t>& shape);
};

}  // namespace tiramisu::nn