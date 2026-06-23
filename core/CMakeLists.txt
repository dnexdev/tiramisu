add_library(tiramisu_core
  src/version.cpp
  src/storage.cpp
  src/tensor.cpp
  src/dtype.cpp
  src/cuda_common.cpp
  src/cuda_memory.cpp
)

target_include_directories(tiramisu_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tiramisu_core
  PUBLIC tiramisu_warnings tiramisu_sanitizers
)

if(TIRAMISU_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(tiramisu_core PRIVATE src/cuda_memory.cu)
  target_compile_definitions(tiramisu_core PUBLIC TIRAMISU_CUDA_ENABLED)
  target_link_libraries(tiramisu_core PUBLIC CUDA::cudart)
  set_target_properties(tiramisu_core PROPERTIES CUDA_STANDARD 17)
endif()