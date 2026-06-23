#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu::autograd {

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& t);
Tensor exp(const Tensor& t);
Tensor log(const Tensor& t);
Tensor relu(const Tensor& t);
Tensor sum(const Tensor& t);
Tensor mean(const Tensor& t);
Tensor matmul(const Tensor& a, const Tensor& b);

void backward(const Tensor& loss);

Tensor softmax(const Tensor& x);
Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps = 1e-5f);

}  // namespace tiramisu::autograd