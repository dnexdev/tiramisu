#pragma once

#include <vector>

#include "tiramisu/core/tensor.hpp"

namespace tiramisu::optim {

float clip_grad_norm(std::vector<Tensor*>& parameters, float max_norm);

}  // namespace tiramisu::optim
