add_library(tiramisu_nn
  src/parameter.cpp
  src/linear.cpp
  src/sequential.cpp
  src/loss.cpp
)

target_include_directories(tiramisu_nn PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(tiramisu_nn PUBLIC tiramisu_core tiramisu_ops tiramisu_autograd)