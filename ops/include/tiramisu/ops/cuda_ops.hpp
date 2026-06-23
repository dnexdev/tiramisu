#pragma once

#include "tiramisu/core/tensor.hpp"

namespace tiramisu::ops::cuda {

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor gelu(const Tensor& t);
Tensor relu(const Tensor& t);
Tensor softmax(const Tensor& x);
Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps);

}  // namespace tiramisu::ops::cuda
