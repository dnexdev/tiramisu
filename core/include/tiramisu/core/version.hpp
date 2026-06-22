#pragma once

namespace tiramisu {

inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;

// Returns "MAJOR.MINOR.PATCH"
const char* version_string();

}  // namespace tiramisu