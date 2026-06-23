#include "tiramisu/ops/embedding.hpp"

#include <stdexcept>

#include "tiramisu/core/device.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include "tiramisu/ops/cuda_ops.hpp"
#endif

namespace tiramisu::ops {

Tensor embedding(const Tensor& weight, const Tensor& indices) {
#ifdef TIRAMISU_CUDA_ENABLED
  if (weight.device() == Device::CUDA || indices.device() == Device::CUDA) {
    return cuda::embedding(weight, indices);
  }
#endif
  if (weight.shape().size() != 2) {
    throw std::invalid_argument("embedding: weight must be 2D");
  }
  if (indices.shape().size() != 2) {
    throw std::invalid_argument("embedding: indices must be 2D");
  }

  const int64_t batch = indices.shape()[0];
  const int64_t seq = indices.shape()[1];
  const int64_t vocab = weight.shape()[0];
  const int64_t dim = weight.shape()[1];

  Tensor c_indices = indices.contiguous();
  Tensor out({batch, seq, dim}, DType::Float32, weight.device());
  const float* w = weight.data<float>();
  const float* idx = c_indices.data<float>();
  float* o = out.data<float>();

  for (int64_t b = 0; b < batch; b++) {
    for (int64_t s = 0; s < seq; s++) {
      const int64_t token = static_cast<int64_t>(idx[b * seq + s]);
      if (token < 0 || token >= vocab) {
        throw std::out_of_range("embedding: token index out of range");
      }
      const float* row = w + token * dim;
      float* out_row = o + (b * seq + s) * dim;
      std::copy(row, row + dim, out_row);
    }
  }
  return out;
}

}  // namespace tiramisu::ops
