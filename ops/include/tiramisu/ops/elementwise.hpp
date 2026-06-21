#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu {
namespace ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);

}
}