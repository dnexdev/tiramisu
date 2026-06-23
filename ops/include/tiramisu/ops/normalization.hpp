#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::ops {

Tensor softmax(const Tensor& x);

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, 
                 float eps=1e-5f);

}