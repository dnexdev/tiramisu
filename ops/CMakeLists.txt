find_package(OpenMP REQUIRED)

add_library(tiramisu_ops
  cpu/broadcast.cpp
  cpu/elementwise.cpp
  cpu/reduce.cpp
  cpu/matmul.cpp
)

target_include_directories(tiramisu_ops PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tiramisu_ops PUBLIC tiramisu_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tiramisu_ops PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(tiramisu_ops PRIVATE -O3 -mavx2 -mfma)