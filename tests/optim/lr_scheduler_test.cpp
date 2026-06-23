#include <gtest/gtest.h>

#include "tiramisu/optim/lr_scheduler.hpp"

TEST(LRSchedulerTest, CosineDecreases) {
  tiramisu::optim::CosineAnnealingLR scheduler(1.0f, 10, 0.0f);
  float first = scheduler.step();
  for (int i = 0; i < 8; ++i) {
    scheduler.step();
  }
  float later = scheduler.current_lr();
  EXPECT_LT(later, first);
}

TEST(LRSchedulerTest, ReachesMinLr) {
  tiramisu::optim::CosineAnnealingLR scheduler(1.0f, 3, 0.1f);
  scheduler.step();
  scheduler.step();
  scheduler.step();
  EXPECT_NEAR(scheduler.current_lr(), 0.1f, 1e-5f);
}
