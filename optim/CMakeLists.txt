add_library(tiramisu_optim
  src/sgd.cpp
  src/adam.cpp
  src/adamw.cpp
  src/grad_clip.cpp
  src/lr_scheduler.cpp
)

target_include_directories(tiramisu_optim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tiramisu_optim PUBLIC tiramisu_core)