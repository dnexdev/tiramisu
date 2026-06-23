#pragma once

#include <vector>

#include "tiramisu/core/tensor.hpp"

namespace tiramisu::optim {

class AdamW {
 public:
  AdamW(const std::vector<Tensor*>& parameters, float lr = 3e-4f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
        float weight_decay = 0.1f);

  void step();
  void zero_grad();
  void set_lr(float lr) { lr_ = lr; }
  float lr() const { return lr_; }

 private:
  std::vector<Tensor*> parameters_;
  std::vector<Tensor> m_;
  std::vector<Tensor> v_;
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  int64_t t_;
};

}  // namespace tiramisu::optim
