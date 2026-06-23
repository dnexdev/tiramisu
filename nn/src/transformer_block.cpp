#include "tiramisu/nn/transformer_block.hpp"

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

TransformerBlock::TransformerBlock(int64_t d_model, int64_t num_heads)
    : ln1_(d_model), mha_(d_model, num_heads), ln2_(d_model), ffn_(d_model) {}

Tensor TransformerBlock::forward(const Tensor& x) {
  Tensor residual = x;
  Tensor h = ln1_.forward(x);
  h = tiramisu::autograd::add(residual, mha_.forward(h));

  residual = h;
  h = ln2_.forward(h);
  h = tiramisu::autograd::add(residual, ffn_.forward(h));
  return h;
}

std::vector<Tensor*> TransformerBlock::parameters() {
  std::vector<Tensor*> params;
  for (Module* child :
       {static_cast<Module*>(&ln1_), static_cast<Module*>(&mha_),
        static_cast<Module*>(&ln2_), static_cast<Module*>(&ffn_)}) {
    auto child_params = child->parameters();
    params.insert(params.end(), child_params.begin(), child_params.end());
  }
  return params;
}

}  // namespace tiramisu::nn
