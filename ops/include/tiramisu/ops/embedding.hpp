#pragma once

#include "tiramisu/core/tensor.hpp"

namespace tiramisu::ops {

Tensor embedding(const Tensor& weight, const Tensor& indices);

}  // namespace tiramisu::ops
