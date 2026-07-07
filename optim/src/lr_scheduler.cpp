#include "tiramisu/optim/lr_scheduler.hpp"

#include <cmath>
#include <numbers>

namespace tiramisu::optim {

CosineAnnealingLR::CosineAnnealingLR(float base_lr, int64_t total_steps,
                                     float min_lr)
    : base_lr_(base_lr),
      min_lr_(min_lr),
      total_steps_(total_steps),
      step_count_(0),
      current_lr_(base_lr) {}

float CosineAnnealingLR::step() {
  // Degenerate schedule (no steps requested): return whatever we last set.
  // Caller gets base_lr on the very first call in this state.
  if (total_steps_ <= 0) {
    return current_lr_;
  }

  step_count_++;
  if (step_count_ >= total_steps_) {
    current_lr_ = min_lr_;
    return current_lr_;
  }

  // Cosine annealing from SGDR (Loshchilov & Hutter, 2016):
  //   https://arxiv.org/abs/1608.03983
  // lr(t) = min_lr + ½(base_lr − min_lr)(1 + cos(π · t/T))
  // Monotonically decreases from base_lr at t=0 to min_lr at t=T.
  const float progress =
      static_cast<float>(step_count_) / static_cast<float>(total_steps_);
  current_lr_ =
      min_lr_ + 0.5f * (base_lr_ - min_lr_) *
          (1.0f + std::cos(std::numbers::pi_v<float> * progress));
  return current_lr_;
}

}  // namespace tiramisu::optim
