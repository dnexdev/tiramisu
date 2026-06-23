#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::nn {

class Parameter : public Tensor {
 public:
  explicit Parameter(const std::vector<int64_t>& shape,
                     Device device = Device::CPU);
};

}  // namespace tiramisu::nn