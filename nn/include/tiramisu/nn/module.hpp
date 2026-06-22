#pragma once
#include <memory>
#include <vector>

#include "tiramisu/core/tensor.hpp"

namespace tiramisu::nn {

class Module {
 public:
  virtual ~Module();

  virtual Tensor forward(const Tensor& x) = 0;

  virtual std::vector<Tensor*> parameters();
};

}  // namespace tiramisu::nn