#pragma once
#include "tiramisu/core/tensor.hpp"
#include <vector>

namespace tiramisu {
namespace ops {

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a,
                                      const std::vector<int64_t>& shape_b);

}
}