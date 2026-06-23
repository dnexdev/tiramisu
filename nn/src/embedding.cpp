#include "tiramisu/nn/embedding.hpp"

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

Embedding::Embedding(int64_t vocab_size, int64_t d_model)
    : weight_({vocab_size, d_model}) {}

Tensor Embedding::forward(const Tensor& token_ids) {
  return tiramisu::autograd::embedding(weight_, token_ids);
}

std::vector<Tensor*> Embedding::parameters() { return {&weight_}; }

}  // namespace tiramisu::nn
