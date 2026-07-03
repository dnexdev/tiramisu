#include "tiramisu/nn/transformer_block.hpp"

#include "tiramisu/autograd/ops.hpp"

namespace tiramisu::nn {

TransformerBlock::TransformerBlock(int64_t d_model, int64_t num_heads,
                                   Device device)
    : ln1_(d_model, 1e-5f, device),
      mha_(d_model, num_heads, true, device),
      ln2_(d_model, 1e-5f, device),
      ffn_(d_model, device) {}

Tensor TransformerBlock::forward(const Tensor& x) {
  Tensor residual = x;
  Tensor h = ln1_.forward(x);
  h = tiramisu::autograd::add(residual, mha_.forward(h));

  residual = h;
  h = ln2_.forward(h);
  h = tiramisu::autograd::add(residual, ffn_.forward(h));
  return h;
}

Tensor TransformerBlock::forward_prefill(const Tensor& x, KVCacheLayer& cache) {
  Tensor residual = x;
  Tensor h = ln1_.forward(x);
  h = tiramisu::autograd::add(residual, mha_.forward_prefill(h, cache));

  residual = h;
  h = ln2_.forward(h);
  h = tiramisu::autograd::add(residual, ffn_.forward(h));
  return h;
}

Tensor TransformerBlock::forward_decode(const Tensor& x, KVCacheLayer& cache) {
  Tensor residual = x;
  Tensor h = ln1_.forward(x);
  h = tiramisu::autograd::add(residual, mha_.forward_decode(h, cache));

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
