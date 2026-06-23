#include <gtest/gtest.h>

#include "tiramisu/ops/broadcast.hpp"

TEST(BroadcastTest, CompatibleShapes) {
  EXPECT_EQ(tiramisu::ops::broadcast_shapes({3, 4}, {3, 4}),
            std::vector<int64_t>({3, 4}));
}

TEST(BroadcastTest, LeadingOnes) {
  EXPECT_EQ(tiramisu::ops::broadcast_shapes({3, 4}, {1}),
            std::vector<int64_t>({3, 4}));
  EXPECT_EQ(tiramisu::ops::broadcast_shapes({3, 4}, {1, 4}),
            std::vector<int64_t>({3, 4}));
}

TEST(BroadcastTest, TrailingOnes) {
  EXPECT_EQ(tiramisu::ops::broadcast_shapes({3, 4}, {3, 1}),
            std::vector<int64_t>({3, 4}));
}

TEST(BroadcastTest, HigherRankLeft) {
  EXPECT_EQ(tiramisu::ops::broadcast_shapes({2, 3, 4}, {4}),
            std::vector<int64_t>({2, 3, 4}));
}

TEST(BroadcastTest, IncompatibleShapesThrow) {
  EXPECT_THROW(tiramisu::ops::broadcast_shapes({3, 4}, {3, 5}),
               std::invalid_argument);
}
