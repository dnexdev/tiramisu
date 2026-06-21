#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::autograd {

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);

void backward(const Tensor& loss);
}