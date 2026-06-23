#include "tiramisu/optim/lr_scheduler.hpp"

#include <cmath>

namespace tiramisu::optim {

CosineAnnealingLR::CosineAnnealingLR(float base_lr, int64_t total_steps,
                                     float min_lr)
    : base_lr_(base_lr),
      min_lr_(min_lr),
      total_steps_(total_steps),
      step_count_(0),
      current_lr_(base_lr) {}

float CosineAnnealingLR::step() {
  if (total_steps_ <= 0) {
    return current_lr_;
  }

  step_count_++;
  if (step_count_ >= total_steps_) {
    current_lr_ = min_lr_;
    return current_lr_;
  }

  const float progress =
      static_cast<float>(step_count_) / static_cast<float>(total_steps_);
  current_lr_ =
      min_lr_ + 0.5f * (base_lr_ - min_lr_) * (1.0f + std::cos(M_PI * progress));
  return current_lr_;
}

}  // namespace tiramisu::optim
