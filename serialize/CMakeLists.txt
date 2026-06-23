add_library(tiramisu_serialize
  src/checkpoint.cpp
)

target_include_directories(tiramisu_serialize PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tiramisu_serialize PUBLIC tiramisu_core tiramisu_nn)
