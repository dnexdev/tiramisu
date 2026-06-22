add_library(tiramisu_optim
  src/sgd.cpp
  src/adam.cpp
)

target_include_directories(tiramisu_optim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tiramisu_optim PUBLIC tiramisu_core)