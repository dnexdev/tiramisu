#pragma once
#include "tiramisu/core/tensor.hpp"
#include <vector>

namespace tiramisu {

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a,
                                      const std::vector<int64_t>& shape_b);

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);

Tensor sum(const Tensor& t);
Tensor mean(const Tensor& t);

Tensor matmul(const Tensor& a, const Tensor& b);

}