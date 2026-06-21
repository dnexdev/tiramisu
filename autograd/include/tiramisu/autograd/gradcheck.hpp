#pragma once
#include <functional>
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::autograd {

bool gradcheck(const std::function<Tensor(const Tensor&)>& f, 
               const Tensor& input,
               double epsilon = 1e-4,
               double tolerance = 1e-3);

}