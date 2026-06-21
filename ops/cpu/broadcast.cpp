#include "tiramisu/ops/broadcast.hpp"
#include <stdexcept>
#include <algorithm>

namespace tiramisu {
namespace ops {
  
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a, 
                                      const std::vector<int64_t>& shape_b) {
  int64_t rank_a = shape_a.size();
  int64_t rank_b = shape_b.size();
  int64_t out_rank = std::max(rank_a, rank_b);

  std::vector<int64_t> out_shape(out_rank);
  
  for (int64_t i = 0; i < out_rank; i++) {
    int64_t dim_a = (i < out_rank - rank_a) ? 1 : shape_a[i - (out_rank - rank_a)];
    int64_t dim_b = (i < out_rank - rank_b) ? 1 : shape_b[i - (out_rank - rank_b)];

    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::invalid_argument("Tensor shapes are not broadcastable.");
    }
    out_shape[i] = std::max(dim_a, dim_b);
  }
  return out_shape;
}

}
}