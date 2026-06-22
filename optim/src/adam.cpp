#include "tiramisu/optim/adam.hpp"
#include <cmath>

namespace tiramisu::optim {

Adam::Adam(const std::vector<Tensor*>& parameters, float lr, float beta1, float beta2, float eps)
  : parameters_(parameters), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {

  for (Tensor* p : parameters_) {
    m_.emplace_back(p->shape());
    v_.emplace_back(p->shape());

    std::fill_n(m_.back().data<float>(), m_.back().numel(), 0.0f);
    std::fill_n(v_.back().data<float>(), v_.back().numel(), 0.0f);
  }
}

void Adam::step() {
  t_ += 1;

  for (size_t i = 0; i < parameters_.size(); i++) {
    Tensor *p = parameters_[i];
    if (!p->requires_grad() || !p->grad()) continue;

    float* p_data = p->data<float>();
    float* g_data = p->grad()->data<float>();
    float* m_data = m_[i].data<float>();
    float* v_data = v_[i].data<float>();

    for (int64_t j = 0; j < p->numel(); j++) {
      float grad = g_data[j];

      m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * grad;

      v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * grad * grad;

      float m_hat = m_data[j] / (1.0f - std::pow(beta1_, static_cast<float>(t_)));

      float v_hat = v_data[j] / (1.0f - std::pow(beta2_, static_cast<float>(t_)));

      p_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    } 
  }
}

void Adam::zero_grad() {
  for (Tensor* p : parameters_) {
    if (p->grad()) {
      std::fill_n(p->grad()->data<float>(), p->grad()->numel(), 0.0f);
    }
  }
}

}