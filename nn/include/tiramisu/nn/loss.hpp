#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::nn {

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);

}