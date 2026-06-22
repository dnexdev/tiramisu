#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b);

}
}  // namespace tiramisu