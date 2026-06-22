#pragma once
#include "tiramisu/core/tensor.hpp"

namespace tiramisu {
namespace ops {

Tensor sum(const Tensor& t);
Tensor mean(const Tensor& t);

}  // namespace ops
}  // namespace tiramisu