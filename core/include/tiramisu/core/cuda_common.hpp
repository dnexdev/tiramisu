#pragma once

#include <stdexcept>
#include <string>

#include "tiramisu/core/device.hpp"

namespace tiramisu {

#ifdef TIRAMISU_CUDA_ENABLED
#define TIRAMISU_CUDA_CHECK(call)                                           \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA error: ") +                \
                               cudaGetErrorString(err));                    \
    }                                                                       \
  } while (0)
#else
#define TIRAMISU_CUDA_CHECK(call) ((void)0)
#endif

bool cuda_available();
void set_device(Device device);
Device current_device();

}  // namespace tiramisu
