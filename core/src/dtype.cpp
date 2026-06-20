#include "tiramisu/core/dtype.hpp"

#include <stdexcept>

namespace tiramisu {

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return 4;
    default:
      throw std::invalid_argument("Unknown DType in dtype_size()");
  }
}  

}