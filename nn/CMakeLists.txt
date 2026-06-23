add_library(tiramisu_nn
  src/parameter.cpp
  src/linear.cpp
  src/sequential.cpp
  src/loss.cpp
  src/module.cpp
  src/layernorm.cpp
  src/embedding.cpp
  src/multi_head_attention.cpp
  src/feed_forward.cpp
  src/transformer_block.cpp
  src/gpt.cpp
)

target_include_directories(tiramisu_nn PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(tiramisu_nn PUBLIC tiramisu_core tiramisu_ops tiramisu_autograd)
