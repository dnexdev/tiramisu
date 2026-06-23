#include "tiramisu/optim/grad_clip.hpp"

#include <cmath>

namespace tiramisu::optim {

float clip_grad_norm(std::vector<Tensor*>& parameters, float max_norm) {
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
