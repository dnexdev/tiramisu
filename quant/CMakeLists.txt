add_library(tiramisu_quant src/quantize.cpp)

target_include_directories(tiramisu_quant PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tiramisu_quant PUBLIC tiramisu_core)
target_link_libraries(tiramisu_quant PRIVATE tiramisu_warnings)
