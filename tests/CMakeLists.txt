add_executable(tiramisu_tests
  core/version_test.cpp
  core/storage_test.cpp
  core/tensor_test.cpp
  ops/ops_test.cpp
  autograd/autograd.cpp
  nn/nn_test.cpp
)

target_link_libraries(tiramisu_tests PRIVATE
  tiramisu_core
  tiramisu_ops
  tiramisu_autograd
  tiramisu_nn
  tiramisu_optim
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tiramisu_tests)