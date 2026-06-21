#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu {
namespace ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor neg(const Tensor& t);
Tensor exp(const Tensor& t);
Tensor log(const Tensor& t);
Tensor relu(const Tensor& t);

}
}