#include "tiramisu/core/cuda_common.hpp"

#include <stdexcept>

#ifdef TIRAMISU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace tiramisu {

bool cuda_available() {
#ifdef TIRAMISU_CUDA_ENABLED
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
  return false;
#endif
}

void set_device(Device device) {
#ifdef TIRAMISU_CUDA_ENABLED
  if (device == Device::CUDA) {
    TIRAMISU_CUDA_CHECK(cudaSetDevice(0));
  }
#else
  if (device == Device::CUDA) {
    throw std::runtime_error("CUDA support not compiled in");
  }
#endif
}

Device current_device() {
#ifdef TIRAMISU_CUDA_ENABLED
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return Device::CPU;
  }
  return Device::CUDA;
#else
  return Device::CPU;
#endif
}

}  // namespace tiramisu
