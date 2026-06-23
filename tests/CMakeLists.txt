add_executable(tiramisu_tests
  # core
  core/version_test.cpp
  core/storage_test.cpp
  core/tensor_test.cpp
  core/tensor_slice_test.cpp
  core/tensor_autograd_test.cpp
  # ops
  ops/broadcast_test.cpp
  ops/elementwise_test.cpp
  ops/reduce_test.cpp
  ops/matmul_test.cpp
  ops/normalization_test.cpp
  # autograd
  autograd/graph_test.cpp
  autograd/gradcheck_test.cpp
  autograd/grad_mode_test.cpp
  autograd/integration_test.cpp
  # nn
  nn/linear_test.cpp
  nn/sequential_test.cpp
  nn/loss_test.cpp
  nn/layernorm_test.cpp
  nn/embedding_test.cpp
  nn/multi_head_attention_test.cpp
  nn/feed_forward_test.cpp
  nn/transformer_block_test.cpp
  nn/gpt_test.cpp
  # optim
  optim/sgd_test.cpp
  optim/adam_test.cpp
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
