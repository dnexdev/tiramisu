add_executable(matmul_bench matmul_bench.cpp)
target_link_libraries(matmul_bench PRIVATE tiramisu_core tiramisu_ops)

# Same codegen flags as tiramisu_ops (ops/CMakeLists.txt), so the naive
# baseline is compiled the same way as the kernel it is compared against.
if(MSVC)
  target_compile_options(matmul_bench PRIVATE /O2 /arch:AVX2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(matmul_bench PRIVATE -O3 -mavx2 -mfma)
else()
  target_compile_options(matmul_bench PRIVATE -O3)
endif()
