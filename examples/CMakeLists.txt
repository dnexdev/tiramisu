if(EMSCRIPTEN)
  add_executable(tiramisu_infer generate_wasm.cpp)
  target_link_libraries(tiramisu_infer PRIVATE
    tiramisu_core
    tiramisu_ops
    tiramisu_autograd
    tiramisu_nn
    tiramisu_serialize
  )
  target_compile_options(tiramisu_infer PRIVATE -O3 -msimd128)
  target_link_options(tiramisu_infer PRIVATE
    "SHELL:-s WASM=1"
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME=createTiramisuModule"
    "SHELL:-s EXPORTED_FUNCTIONS=['_tiramisu_init','_tiramisu_generate','_tiramisu_generate_begin','_tiramisu_generate_step','_tiramisu_generate_end','_tiramisu_free','_malloc','_free']"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString']"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s INITIAL_MEMORY=67108864"
    "SHELL:-O3"
  )

  add_executable(mnist_infer mnist_infer_wasm.cpp)
  target_link_libraries(mnist_infer PRIVATE
    tiramisu_core
    tiramisu_ops
    tiramisu_autograd
    tiramisu_nn
    tiramisu_serialize
  )
  target_compile_options(mnist_infer PRIVATE -O3 -msimd128)
  target_link_options(mnist_infer PRIVATE
    "SHELL:-s WASM=1"
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME=createMnistModule"
    "SHELL:-s EXPORTED_FUNCTIONS=['_mnist_init','_mnist_predict','_mnist_predict_ex','_mnist_free','_malloc','_free']"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s INITIAL_MEMORY=8388608"
    "SHELL:-O3"
  )
else()
  add_executable(hello_tiramisu hello_tiramisu.cpp)
  add_executable(mnist mnist.cpp)
  add_executable(train_shakespeare train_shakespeare.cpp)
  add_executable(quantize_checkpoint quantize_checkpoint.cpp)
  target_link_libraries(hello_tiramisu PRIVATE tiramisu_core)
  target_link_libraries(mnist PRIVATE tiramisu_core tiramisu_ops tiramisu_autograd tiramisu_nn tiramisu_optim tiramisu_serialize)
  target_link_libraries(train_shakespeare PRIVATE tiramisu_core tiramisu_ops tiramisu_autograd tiramisu_nn tiramisu_optim tiramisu_serialize)
  target_link_libraries(quantize_checkpoint PRIVATE tiramisu_core tiramisu_nn tiramisu_serialize tiramisu_quant)
endif()
