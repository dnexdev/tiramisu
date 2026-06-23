find_package(OpenMP REQUIRED)

set(TIRAMISU_OPS_CPU_SOURCES
  cpu/broadcast.cpp
  cpu/elementwise.cpp
  cpu/reduce.cpp
  cpu/matmul.cpp
  cpu/normalization.cpp
  cpu/embedding.cpp
)

add_library(tiramisu_ops ${TIRAMISU_OPS_CPU_SOURCES})

target_include_directories(tiramisu_ops PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tiramisu_ops PUBLIC tiramisu_core)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tiramisu_ops PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(tiramisu_ops PRIVATE -O3 -mavx2 -mfma)

if(TIRAMISU_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(tiramisu_ops PRIVATE cuda/ops.cu cuda/backward.cu)
  target_compile_definitions(tiramisu_ops PUBLIC TIRAMISU_CUDA_ENABLED)
  target_link_libraries(tiramisu_ops PUBLIC CUDA::cudart)
  set_target_properties(tiramisu_ops PROPERTIES CUDA_STANDARD 17)
endif()
