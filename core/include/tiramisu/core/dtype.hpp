#pragma once
#include <cstddef>

namespace tiramisu {

enum class DType {
  Float32,
};

std::size_t dtype_size(DType dtype);

}