#include "tiramisu/optim/grad_clip.hpp"

#include <cmath>
#include <stdexcept>

#include "tiramisu/core/device.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include "tiramisu/ops/cuda_ops.hpp"
#endif

namespace tiramisu::optim {

float clip_grad_norm(std::vector<Tensor*>& parameters, float max_norm) {
#ifdef TIRAMISU_CUDA_ENABLED
  bool any_cuda = false;
  bool any_cpu = false;
  for (Tensor* p : parameters) {
    if (!p->grad()) continue;
    if (p->grad()->device() == Device::CUDA) {
      any_cuda = true;
    } else {
      any_cpu = true;
    }
  }
  if (any_cuda && any_cpu) {
    throw std::runtime_error(
        "clip_grad_norm: mixed CPU/CUDA gradients are not supported");
  }
  if (any_cuda) {
    return ops::cuda::clip_grad_norm(parameters, max_norm);
  }
#endif

  double total_norm_sq = 0.0;
  for (Tensor* p : parameters) {
    if (!p->grad()) continue;
    const float* g = p->grad()->data<float>();
    for (int64_t i = 0; i < p->grad()->numel(); ++i) {
      total_norm_sq += static_cast<double>(g[i]) * g[i];
    }
  }

  const float total_norm = static_cast<float>(std::sqrt(total_norm_sq));
  if (total_norm <= max_norm || total_norm == 0.0f) {
    return total_norm;
  }

  const float scale = max_norm / total_norm;
  for (Tensor* p : parameters) {
    if (!p->grad()) continue;
    float* g = p->grad()->data<float>();
    for (int64_t i = 0; i < p->grad()->numel(); ++i) {
      g[i] *= scale;
    }
  }
  return total_norm;
}

}  // namespace tiramisu::optim
