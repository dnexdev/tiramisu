add_library(tiramisu_autograd
  src/grad_mode.cpp
  src/gradcheck.cpp
  src/ops.cpp
)

target_include_directories(tiramisu_autograd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include
)

target_link_libraries(tiramisu_autograd PUBLIC tiramisu_core tiramisu_ops)