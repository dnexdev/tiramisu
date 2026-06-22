#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/core/node.hpp"
#include "tiramisu/ops/elementwise.hpp"
#include "tiramisu/ops/reduce.hpp"
#include "tiramisu/ops/matmul.hpp"

#include <unordered_set>
#include <algorithm>

namespace tiramisu::autograd {

// sum grad_output down to target_shape by summing over any leading
// dims that were broadcast
// ex: grad_output (64, 10), target_shape (10,) -> sum over dim 0 -> (10,)
static Tensor reduce_grad_to(const Tensor& grad, const std::vector<int64_t>& target_shape) {
  if (grad.shape() == target_shape) {
    return grad;
  }

  int64_t extra_dims = (int64_t)grad.shape().size() - (int64_t)target_shape.size();

  Tensor result = grad.contiguous();

  for (int64_t d = 0; d < extra_dims; d++) {
    int64_t rows = result.shape()[0];
    int64_t cols = result.numel() / rows;

    Tensor summed({cols});

    float* src = result.data<float>();
    float* dst = summed.data<float>();
    std::fill_n(dst, cols, 0.0f);

    for (int64_t r = 0; r < rows; r++) {
      for (int64_t c = 0; c < cols; c++) {
        dst[c] += src[r * cols + c];
      }
    }
    result = summed;
  }
  return result;
}

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::add(a, b);

  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      return std::vector<Tensor>{
        reduce_grad_to(grad_output, a.shape()),
        reduce_grad_to(grad_output, b.shape())
      };
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

Tensor sub(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::sub(a, b);
  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      return std::vector<Tensor>{
        reduce_grad_to(grad_output, a.shape()),
        reduce_grad_to(tiramisu::ops::neg(grad_output), b.shape())
      };
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::div(a, b);
  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      // grad_a = grad_output / b
      Tensor grad_a = tiramisu::ops::div(grad_output, b);

      // grad_b = -grad_output * a / (b * b)
      Tensor b_sq = tiramisu::ops::mul(b, b);
      Tensor neg_grad = tiramisu::ops::neg(grad_output);
      Tensor num = tiramisu::ops::mul(neg_grad, a);
      Tensor grad_b = tiramisu::ops::div(num, b_sq);

      return std::vector<Tensor>{grad_a, grad_b};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor neg(const Tensor& t) {
  Tensor out = tiramisu::ops::neg(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    node->backward_fn = [](const Tensor& grad_output) {
      // grad_t = -grad_output
      return std::vector<Tensor>{tiramisu::ops::neg(grad_output)};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor exp(const Tensor& t) {
  Tensor out = tiramisu::ops::exp(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    Tensor out_val(out.shape(), out.dtype(), out.device());
    std::copy(out.data<float>(), out.data<float>() + out.numel(), out_val.data<float>());
    node->backward_fn = [out_val](const Tensor& grad_output) {
      // grad_t = grad_output * (detached out)
      return std::vector<Tensor>{tiramisu::ops::mul(grad_output, out_val)};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor log(const Tensor& t) {
  Tensor out = tiramisu::ops::log(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    node->backward_fn = [t](const Tensor& grad_output) {
      return std::vector<Tensor>{tiramisu::ops::div(grad_output, t)};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor relu(const Tensor& t) {
  Tensor out = tiramisu::ops::relu(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    node->backward_fn = [t](const Tensor& grad_output) {
      Tensor t_c = t.contiguous();
      Tensor g_c = grad_output.contiguous();
      Tensor grad_t(t_c.shape(), t_c.dtype(), t_c.device());

      const float* t_data = t_c.data<float>();
      const float* g_data = g_c.data<float>();
      float *gt_data = grad_t.data<float>();

      int64_t n = t_c.numel();
      for (int64_t i = 0; i < n; i++) {
        gt_data[i] = t_data[i] > 0.0f ? g_data[i] : 0.0f;
      }

      return std::vector<Tensor>{grad_t};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor sum(const Tensor& t) {
  Tensor out = tiramisu::ops::sum(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    auto t_shape = t.shape();
    node->backward_fn = [t_shape](const Tensor& grad_output) {
      Tensor grad_t(t_shape, grad_output.dtype(), grad_output.device());
      float g_val = grad_output.data<float>()[0];
      std::fill_n(grad_t.data<float>(), grad_t.numel(), g_val);

      return std::vector<Tensor>{grad_t};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor mean(const Tensor& t) {
  Tensor out = tiramisu::ops::mean(t);
  if (grad_enabled() && t.requires_grad()) {
    auto node = std::make_shared<Node>();
    node->inputs = {t};
    auto t_shape = t.shape();
    auto n = t.numel();
    node->backward_fn = [t_shape, n](const Tensor& grad_output) {
      Tensor grad_t(t_shape, grad_output.dtype(), grad_output.device());

      float g_val = grad_output.data<float>()[0] / static_cast<float>(n);
      std::fill_n(grad_t.data<float>(), grad_t.numel(), g_val);

      return std::vector<Tensor>{grad_t};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  Tensor out = tiramisu::ops::matmul(a, b);
  if (grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto node = std::make_shared<Node>();
    node->inputs = {a, b};
    node->backward_fn = [a, b](const Tensor& grad_output) {
      Tensor grad_a = tiramisu::ops::matmul(grad_output, b.transpose(0, 1));

      Tensor grad_b = tiramisu::ops::matmul(a.transpose(0, 1), grad_output);
      return std::vector<Tensor>{grad_a, grad_b};
    };
    out.set_requires_grad(true);
    out.set_grad_fn(node);
  }
  return out;
}

}