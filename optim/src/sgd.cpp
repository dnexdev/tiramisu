#include "tiramisu/optim/sgd.hpp"

namespace tiramisu::optim {

SGD::SGD(const std::vector<Tensor*>& parameters, float lr)
  : parameters_(parameters), lr_(lr) {}

void SGD::step() {
  for (Tensor* p : parameters_) {
    if (!p->requires_grad() || !p->grad()) continue;

    float* p_data = p->data<float>();
    float* g_data = p->grad()->data<float>();

    for (int64_t i = 0; i < p->numel(); i++) {
      p_data[i] -= lr_ * g_data[i];
    }
  }
}

void SGD::zero_grad() {
  for (Tensor* p : parameters_) {
    if (p->grad()) {
      std::fill_n(p->grad()->data<float>(), p->grad()->numel(), 0.0f);
    }
  }
}

}