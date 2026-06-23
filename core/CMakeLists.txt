add_library(tiramisu_core
  src/version.cpp
  src/storage.cpp
  src/tensor.cpp
  src/dtype.cpp
  src/cuda_common.cpp
)

target_include_directories(tiramisu_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tiramisu_core
  PUBLIC tiramisu_warnings tiramisu_sanitizers
)