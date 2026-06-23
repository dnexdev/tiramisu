#pragma once

#include <cstdint>

namespace tiramisu::optim {

class CosineAnnealingLR {
 public:
  CosineAnnealingLR(float base_lr, int64_t total_steps, float min_lr = 0.0f);

  float step();
  float current_lr() const { return current_lr_; }

 private:
  float base_lr_;
  float min_lr_;
  int64_t total_steps_;
  int64_t step_count_;
  float current_lr_;
};

}  // namespace tiramisu::optim
