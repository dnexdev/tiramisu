include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_C bindings.cpp)

target_link_libraries(_C PRIVATE
  tiramisu_core
  tiramisu_ops
  tiramisu_autograd
  tiramisu_nn
  tiramisu_optim
)

target_compile_options(_C PRIVATE -Wno-pedantic)

install(TARGETS _C LIBRARY DESTINATION tiramisu)
