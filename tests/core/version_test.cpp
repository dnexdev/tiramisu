#include <gtest/gtest.h>

#include "tiramisu/core/version.hpp"

TEST(Version, ReturnsExpectedString) {
  EXPECT_STREQ(tiramisu::version_string(), "0.1.0");
}

TEST(Version, ConstantsMatchString) {
  EXPECT_EQ(tiramisu::kVersionMajor, 0);
  EXPECT_EQ(tiramisu::kVersionMinor, 1);
  EXPECT_EQ(tiramisu::kVersionPatch, 0);
}