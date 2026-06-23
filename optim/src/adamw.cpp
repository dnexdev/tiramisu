#include "tiramisu/optim/adamw.hpp"

#include <cmath>

#include "tiramisu/core/device.hpp"

#ifdef TIRAMISU_CUDA_ENABLED
#include "tiramisu/ops/cuda_ops.hpp"
#endif

namespace tiramisu::optim {

AdamW::AdamW(const std::vector<Tensor*>& parameters, float lr, float beta1,
             float beta2, float eps, float weight_decay)
    : parameters_(parameters),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      t_(0) {
  for (Tensor* p : parameters_) {
    const Device dev = p->device();
    m_.emplace_back(p->shape(), DType::Float32, dev);
    v_.emplace_back(p->shape(), DType::Float32, dev);
#ifdef TIRAMISU_CUDA_ENABLED
    if (dev == Device::CUDA) {
      ops::cuda::zero_grad(m_.back().data<float>(), m_.back().numel());
      ops::cuda::zero_grad(v_.back().data<float>(), v_.back().numel());
      continue;
    }
#endif
    std::fill_n(m_.back().data<float>(), m_.back().numel(), 0.0f);
    std::fill_n(v_.back().data<float>(), v_.back().numel(), 0.0f);
  }
}

void AdamW::step() {
  t_ += 1;

  for (size_t i = 0; i < parameters_.size(); i++) {
    Tensor* p = parameters_[i];
    if (!p->requires_grad() || !p->grad()) {
      continue;
    }

#ifdef TIRAMISU_CUDA_ENABLED
    if (p->device() == Device::CUDA) {
      ops::cuda::adamw_step(p->data<float>(), p->grad()->data<float>(),
                            m_[i].data<float>(), v_[i].data<float>(),
                            p->numel(), lr_, beta1_, beta2_, eps_,
                            weight_decay_, t_);
      continue;
    }
#endif

    float* p_data = p->data<float>();
    float* g_data = p->grad()->data<float>();
    float* m_data = m_[i].data<float>();
    float* v_data = v_[i].data<float>();

    for (int64_t j = 0; j < p->numel(); j++) {
      const float grad = g_data[j];

      m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * grad;
      v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * grad * grad;

      const float m_hat =
          m_data[j] / (1.0f - std::pow(beta1_, static_cast<float>(t_)));
      const float v_hat =
          v_data[j] / (1.0f - std::pow(beta2_, static_cast<float>(t_)));

      p_data[j] -= lr_ * (m_hat / (std::sqrt(v_hat) + eps_) +
                          weight_decay_ * p_data[j]);
    }
  }
}

void AdamW::zero_grad() {
  for (Tensor* p : parameters_) {
    if (!p->grad()) {
      continue;
    }
#ifdef TIRAMISU_CUDA_ENABLED
    if (p->device() == Device::CUDA) {
      ops::cuda::zero_grad(p->grad()->data<float>(), p->grad()->numel());
      continue;
    }
#endif
    std::fill_n(p->grad()->data<float>(), p->grad()->numel(), 0.0f);
  }
}

}  // namespace tiramisu::optim
