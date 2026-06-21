#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/core/node.hpp"
#include "tiramisu/ops/elementwise.hpp"

#include <unordered_set>
#include <algorithm>

namespace tiramisu::autograd {

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::add(a, b);

  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [](const Tensor& grad_output) {
      return std::vector<Tensor>{grad_output, grad_output};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::mul(a, b);

  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      return std::vector<Tensor>{
        tiramisu::ops::mul(grad_output, b),
        tiramisu::ops::mul(grad_output, a)
      };
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

void backward(const Tensor& loss) {
  std::vector<Tensor> topo;
  std::unordered_set<Node*> visited;

  std::function<void(const Tensor&)> build_topo = [&](const Tensor& t) {
    auto n = t.grad_fn();
    if (!n || visited.count(n.get())) return;

    visited.insert(n.get());
    for (const auto& input : n->inputs) {
      build_topo(input);
    }
    topo.push_back(t);
  };

  build_topo(loss);

  Tensor ones_grad(loss.shape(), loss.dtype(), loss.device());
  std::fill_n(ones_grad.data<float>(), ones_grad.numel(), 1.0f);
  Tensor loss_mut = loss;
  loss_mut.accumulate_grad(ones_grad);

  NoGradGuard guard;

  for (auto it = topo.rbegin(); it != topo.rend(); it++) {
    Tensor& t = *it;
    auto node = t.grad_fn();
    if (!node || !t.grad()) continue;
    auto grads = node->backward_fn(*t.grad());

    for (size_t i = 0; i < node->inputs.size(); i++) {
      if (node->inputs[i].requires_grad()) {
        Tensor input_mut = node->inputs[i];
        input_mut.accumulate_grad(grads[i]);
      }
    }
  }
}

}